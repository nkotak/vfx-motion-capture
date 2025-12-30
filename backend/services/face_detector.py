"""
Face detection and recognition service using InsightFace.
Handles face detection, alignment, embedding extraction, and face swapping support.
"""

import asyncio
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import cv2
from PIL import Image
from loguru import logger

from backend.core.config import settings
from backend.core.models import FaceData
from backend.core.exceptions import FaceDetectionError, NoFaceDetectedError


@dataclass
class DetectedFace:
    """Information about a detected face."""

    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    landmarks: np.ndarray  # 5-point or 68-point landmarks
    confidence: float
    embedding: Optional[np.ndarray] = None  # Face embedding for recognition
    age: Optional[int] = None
    gender: Optional[str] = None
    aligned_face: Optional[np.ndarray] = None  # Aligned face crop

    @property
    def center(self) -> Tuple[float, float]:
        """Get face center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def size(self) -> Tuple[float, float]:
        """Get face width and height."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)

    @property
    def area(self) -> float:
        """Get face area."""
        w, h = self.size
        return w * h


@dataclass
class FaceSwapData:
    """Data needed for face swapping."""

    source_face: DetectedFace
    target_faces: List[DetectedFace]
    source_embedding: np.ndarray
    aimm_face: Optional[Any] = None  # For InsightFace swapper


class FaceDetector:
    """
    Face detection and analysis service.

    Features:
    - Face detection with bounding boxes
    - Facial landmark detection (5 or 68 points)
    - Face embedding extraction for recognition
    - Face alignment and cropping
    - Age/gender estimation
    - Face swap preparation
    """

    def __init__(
        self,
        detection_threshold: float = 0.5,
        device: str = None,
        det_model: str = "buffalo_l",
    ):
        """
        Initialize face detector.

        Args:
            detection_threshold: Minimum confidence for detection
            device: "cuda" or "cpu"
            det_model: InsightFace model name ("buffalo_l", "buffalo_sc", etc.)
        """
        self.detection_threshold = detection_threshold
        self.device = device or settings.device
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.det_model = det_model
        self._model = None
        self._swapper = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the face detection model."""
        if self._initialized:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis

            # Determine providers based on device
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif self.device == "mps":
                # CoreML provider might not be available in all builds, fallback to CPU if needed
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            # Initialize face analysis
            self._model = FaceAnalysis(
                name=self.det_model,
                providers=providers,
            )
            self._model.prepare(ctx_id=0 if self.device == "cuda" else -1)

            self._initialized = True
            logger.info(f"Initialized face detector with {self.det_model} model")

        except ImportError:
            raise FaceDetectionError(
                "InsightFace not installed. Run: pip install insightface onnxruntime-gpu"
            )
        except Exception as e:
            raise FaceDetectionError(f"Failed to initialize face detector: {e}")

    async def detect_faces(
        self,
        image: np.ndarray | Path | Image.Image,
        max_faces: int = 10,
        min_face_size: int = 20,
        extract_embedding: bool = True,
    ) -> List[DetectedFace]:
        """
        Detect faces in an image.

        Args:
            image: Input image
            max_faces: Maximum number of faces to detect
            min_face_size: Minimum face size in pixels
            extract_embedding: Whether to extract face embeddings

        Returns:
            List of DetectedFace objects
        """
        await self.initialize()

        # Convert to numpy array
        if isinstance(image, Path):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Detect faces
        faces_raw = await asyncio.to_thread(self._model.get, image)

        if not faces_raw:
            return []

        # Filter and convert
        faces = []
        for face in faces_raw:
            # Filter by size
            bbox = face.bbox.astype(float)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width < min_face_size or height < min_face_size:
                continue

            # Filter by confidence
            if face.det_score < self.detection_threshold:
                continue

            detected = DetectedFace(
                bbox=tuple(bbox),
                landmarks=face.kps if hasattr(face, "kps") else face.landmark_2d_106,
                confidence=float(face.det_score),
                embedding=face.embedding if extract_embedding and hasattr(face, "embedding") else None,
                age=int(face.age) if hasattr(face, "age") else None,
                gender="male" if hasattr(face, "gender") and face.gender == 1 else "female" if hasattr(face, "gender") else None,
            )

            faces.append(detected)

            if len(faces) >= max_faces:
                break

        # Sort by face area (largest first)
        faces.sort(key=lambda f: f.area, reverse=True)

        return faces

    async def get_primary_face(
        self,
        image: np.ndarray | Path | Image.Image,
    ) -> DetectedFace:
        """
        Get the primary (largest) face in an image.

        Raises NoFaceDetectedError if no face is found.
        """
        faces = await self.detect_faces(image, max_faces=1)
        if not faces:
            raise NoFaceDetectedError("image")
        return faces[0]

    async def align_face(
        self,
        image: np.ndarray,
        face: DetectedFace,
        output_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        Align and crop a face from an image.

        Args:
            image: Source image
            face: Detected face to align
            output_size: Output image size

        Returns:
            Aligned face image
        """
        from insightface.utils import face_align

        # Use 5-point landmarks for alignment
        landmarks = face.landmarks
        if landmarks.shape[0] > 5:
            # Extract 5-point landmarks from larger set
            # Standard 5 points: left_eye, right_eye, nose, left_mouth, right_mouth
            landmarks = landmarks[:5]

        aligned = await asyncio.to_thread(
            face_align.norm_crop,
            image,
            landmarks,
            image_size=output_size[0],
        )

        return aligned

    async def extract_embedding(
        self,
        image: np.ndarray | Path | Image.Image,
    ) -> np.ndarray:
        """
        Extract face embedding from an image.

        Returns a 512-dimensional embedding vector.
        """
        face = await self.get_primary_face(image)
        if face.embedding is None:
            raise FaceDetectionError("Failed to extract face embedding")
        return face.embedding

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two face embeddings.

        Returns:
            Similarity score between -1 and 1 (higher = more similar)
        """
        # Normalize embeddings
        e1 = embedding1 / np.linalg.norm(embedding1)
        e2 = embedding2 / np.linalg.norm(embedding2)

        # Compute cosine similarity
        return float(np.dot(e1, e2))

    async def detect_faces_in_video(
        self,
        video_path: Path,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[List[DetectedFace]]:
        """
        Detect faces in all frames of a video.

        Args:
            video_path: Path to video file
            fps: Target FPS for processing
            max_frames: Maximum frames to process
            progress_callback: Progress callback function

        Returns:
            List of face lists for each frame
        """
        await self.initialize()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FaceDetectionError(f"Could not open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            target_fps = fps or video_fps
            frame_skip = max(1, int(video_fps / target_fps))

            all_faces = []
            frame_idx = 0
            processed = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect faces
                    faces = await self.detect_faces(
                        frame_rgb,
                        extract_embedding=False,  # Skip for speed
                    )
                    all_faces.append(faces)

                    processed += 1

                    if progress_callback:
                        progress = (frame_idx / total_frames) * 100
                        progress_callback(progress, f"Detecting faces: frame {processed}")

                    if max_frames and processed >= max_frames:
                        break

                frame_idx += 1

                # Yield control periodically
                if frame_idx % 30 == 0:
                    await asyncio.sleep(0)

            logger.info(f"Detected faces in {processed} frames")
            return all_faces

        finally:
            cap.release()

    async def prepare_face_swap(
        self,
        source_image: np.ndarray | Path | Image.Image,
        target_image: np.ndarray | Path | Image.Image,
    ) -> FaceSwapData:
        """
        Prepare data for face swapping.

        Args:
            source_image: Image containing the source face
            target_image: Image containing target face(s) to replace

        Returns:
            FaceSwapData with source and target face information
        """
        # Detect source face
        source_face = await self.get_primary_face(source_image)
        if source_face.embedding is None:
            # Re-detect with embedding
            if isinstance(source_image, (Path, str)):
                source_image = cv2.imread(str(source_image))
                source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            elif isinstance(source_image, Image.Image):
                source_image = np.array(source_image)

            faces = await self.detect_faces(source_image, max_faces=1, extract_embedding=True)
            if not faces or faces[0].embedding is None:
                raise FaceDetectionError("Could not extract source face embedding")
            source_face = faces[0]

        # Detect target faces
        target_faces = await self.detect_faces(target_image, max_faces=10, extract_embedding=False)
        if not target_faces:
            raise NoFaceDetectedError("target image")

        return FaceSwapData(
            source_face=source_face,
            target_faces=target_faces,
            source_embedding=source_face.embedding,
        )

    def draw_faces(
        self,
        image: np.ndarray,
        faces: List[DetectedFace],
        draw_bbox: bool = True,
        draw_landmarks: bool = True,
        draw_info: bool = True,
    ) -> np.ndarray:
        """
        Draw detected faces on an image.

        Args:
            image: Input image
            faces: List of detected faces
            draw_bbox: Draw bounding boxes
            draw_landmarks: Draw facial landmarks
            draw_info: Draw age/gender info

        Returns:
            Image with face annotations
        """
        output = image.copy()

        for i, face in enumerate(faces):
            # Draw bounding box
            if draw_bbox:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw landmarks
            if draw_landmarks and face.landmarks is not None:
                for point in face.landmarks:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(output, (x, y), 2, (255, 0, 0), -1)

            # Draw info
            if draw_info:
                x1, y1 = int(face.bbox[0]), int(face.bbox[1])
                info_parts = [f"#{i+1}"]
                if face.age:
                    info_parts.append(f"Age:{face.age}")
                if face.gender:
                    info_parts.append(face.gender[0].upper())
                info_parts.append(f"{face.confidence:.2f}")

                info_text = " ".join(info_parts)
                cv2.putText(
                    output,
                    info_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        return output

    def close(self) -> None:
        """Release resources."""
        if self._model is not None:
            self._model = None
        if self._swapper is not None:
            self._swapper = None
        self._initialized = False


# Singleton instance
_detector: Optional[FaceDetector] = None


def get_face_detector() -> FaceDetector:
    """Get the global face detector instance."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector
