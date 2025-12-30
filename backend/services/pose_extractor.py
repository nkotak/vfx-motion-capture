"""
Pose extraction service using DWPose/MediaPipe.
Extracts body and hand keypoints from images and videos.
"""

import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import cv2
from PIL import Image
from loguru import logger

from backend.core.config import settings
from backend.core.models import PoseData
from backend.core.exceptions import PoseExtractionError, NoPoseDetectedError


@dataclass
class PoseKeypoints:
    """Pose keypoints for a detected person."""

    body: np.ndarray  # Shape: (N, 3) for N keypoints with (x, y, confidence)
    left_hand: Optional[np.ndarray] = None  # Shape: (21, 3)
    right_hand: Optional[np.ndarray] = None  # Shape: (21, 3)
    face: Optional[np.ndarray] = None  # Shape: (68, 3) or (478, 3) for MediaPipe
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    confidence: float = 0.0


@dataclass
class PoseSequence:
    """Pose data for a sequence of frames."""

    frames: List[List[PoseKeypoints]] = field(default_factory=list)
    fps: float = 24.0
    width: int = 0
    height: int = 0

    def __len__(self) -> int:
        return len(self.frames)

    def get_primary_poses(self) -> List[Optional[PoseKeypoints]]:
        """Get the primary (highest confidence) pose for each frame."""
        return [
            max(frame_poses, key=lambda p: p.confidence) if frame_poses else None
            for frame_poses in self.frames
        ]


class PoseExtractor:
    """
    Pose extraction service supporting multiple backends.

    Backends:
    - DWPose: High accuracy, GPU accelerated
    - MediaPipe: CPU friendly, good for real-time
    - OpenPose: Classic, widely supported
    """

    # COCO 18-point body keypoint names
    BODY_KEYPOINTS = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
        "left_eye", "right_ear", "left_ear"
    ]

    def __init__(
        self,
        backend: str = "mediapipe",
        device: str = None,
        model_complexity: int = 1,
    ):
        """
        Initialize pose extractor.

        Args:
            backend: "dwpose", "mediapipe", or "openpose"
            device: "cuda" or "cpu"
            model_complexity: Model complexity (0, 1, or 2 for MediaPipe)
        """
        self.backend = backend.lower()
        self.device = device or settings.device
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.model_complexity = model_complexity
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the pose detection model."""
        if self._initialized:
            return

        if self.backend == "mediapipe":
            await self._init_mediapipe()
        elif self.backend == "dwpose":
            await self._init_dwpose()
        else:
            raise PoseExtractionError(f"Unknown backend: {self.backend}")

        self._initialized = True
        logger.info(f"Initialized pose extractor with {self.backend} backend")

    async def _init_mediapipe(self) -> None:
        """Initialize MediaPipe Pose."""
        try:
            import mediapipe as mp

            self._mp_pose = mp.solutions.pose
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils

            # Initialize pose detector
            self._model = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # Initialize hand detector
            self._hand_model = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        except ImportError:
            raise PoseExtractionError("MediaPipe not installed. Run: pip install mediapipe")

    async def _init_dwpose(self) -> None:
        """Initialize DWPose (requires controlnet_aux)."""
        try:
            # DWPose is typically used via ComfyUI's controlnet_aux
            # For standalone usage, we'll use a simplified approach
            from controlnet_aux import DWposeDetector

            self._model = DWposeDetector()
            logger.info("DWPose initialized")

        except ImportError:
            logger.warning("DWPose not available, falling back to MediaPipe")
            self.backend = "mediapipe"
            await self._init_mediapipe()

    async def extract_from_image(
        self,
        image: np.ndarray | Path | Image.Image,
        detect_hands: bool = True,
        detect_face: bool = False,
    ) -> List[PoseKeypoints]:
        """
        Extract poses from a single image.

        Args:
            image: Input image (numpy array, path, or PIL Image)
            detect_hands: Whether to detect hand keypoints
            detect_face: Whether to detect face keypoints

        Returns:
            List of detected poses
        """
        await self.initialize()

        # Convert to numpy array
        if isinstance(image, Path):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        if self.backend == "mediapipe":
            return await self._extract_mediapipe(image, detect_hands, detect_face)
        elif self.backend == "dwpose":
            return await self._extract_dwpose(image)
        else:
            raise PoseExtractionError(f"Unknown backend: {self.backend}")

    async def _extract_mediapipe(
        self,
        image: np.ndarray,
        detect_hands: bool,
        detect_face: bool,
    ) -> List[PoseKeypoints]:
        """Extract poses using MediaPipe."""
        height, width = image.shape[:2]

        # Process image
        results = await asyncio.to_thread(self._model.process, image)

        if not results.pose_landmarks:
            return []

        poses = []

        # Extract body keypoints
        landmarks = results.pose_landmarks.landmark
        body_keypoints = np.array([
            [lm.x * width, lm.y * height, lm.visibility]
            for lm in landmarks
        ])

        # Calculate bounding box
        valid_points = body_keypoints[body_keypoints[:, 2] > 0.5]
        if len(valid_points) > 0:
            x1, y1 = valid_points[:, :2].min(axis=0)
            x2, y2 = valid_points[:, :2].max(axis=0)
            # Add padding
            pad = 20
            bbox = (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(width, x2 + pad),
                min(height, y2 + pad),
            )
        else:
            bbox = None

        # Calculate overall confidence
        confidence = float(np.mean(body_keypoints[:, 2]))

        pose = PoseKeypoints(
            body=body_keypoints,
            bbox=bbox,
            confidence=confidence,
        )

        # Extract hands if requested
        if detect_hands:
            hand_results = await asyncio.to_thread(self._hand_model.process, image)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness
                ):
                    hand_kpts = np.array([
                        [lm.x * width, lm.y * height, 1.0]
                        for lm in hand_landmarks.landmark
                    ])
                    hand_label = handedness.classification[0].label
                    if hand_label == "Left":
                        pose.left_hand = hand_kpts
                    else:
                        pose.right_hand = hand_kpts

        poses.append(pose)
        return poses

    async def _extract_dwpose(self, image: np.ndarray) -> List[PoseKeypoints]:
        """Extract poses using DWPose."""
        # DWPose returns a processed image, we need to extract keypoints
        result = await asyncio.to_thread(self._model, image)

        # The result format depends on the DWPose version
        # This is a simplified implementation
        poses = []

        # If result contains poses data
        if hasattr(result, "bodies") and result.bodies:
            for body in result.bodies:
                pose = PoseKeypoints(
                    body=np.array(body.keypoints),
                    confidence=float(np.mean([kp[2] for kp in body.keypoints if len(kp) > 2])),
                )
                poses.append(pose)

        return poses

    async def extract_from_video(
        self,
        video_path: Path,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        detect_hands: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> PoseSequence:
        """
        Extract poses from a video file.

        Args:
            video_path: Path to video file
            fps: Target FPS for extraction (None = use video's FPS)
            max_frames: Maximum frames to process
            detect_hands: Whether to detect hand keypoints
            progress_callback: Optional callback for progress updates

        Returns:
            PoseSequence containing poses for all frames
        """
        await self.initialize()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise PoseExtractionError(f"Could not open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            target_fps = fps or video_fps
            frame_skip = max(1, int(video_fps / target_fps))

            sequence = PoseSequence(fps=target_fps, width=width, height=height)

            frame_idx = 0
            processed = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Extract poses
                    poses = await self.extract_from_image(
                        frame_rgb,
                        detect_hands=detect_hands,
                        detect_face=False,
                    )
                    sequence.frames.append(poses)

                    processed += 1

                    if progress_callback:
                        progress = (frame_idx / total_frames) * 100
                        progress_callback(progress, f"Extracting poses: frame {processed}")

                    if max_frames and processed >= max_frames:
                        break

                frame_idx += 1

                # Yield control periodically
                if frame_idx % 30 == 0:
                    await asyncio.sleep(0)

            logger.info(f"Extracted poses from {processed} frames")
            return sequence

        finally:
            cap.release()

    def render_pose(
        self,
        image: np.ndarray,
        pose: PoseKeypoints,
        draw_body: bool = True,
        draw_hands: bool = True,
        line_thickness: int = 2,
        point_radius: int = 4,
    ) -> np.ndarray:
        """
        Render pose keypoints on an image.

        Args:
            image: Input image
            pose: Pose keypoints to render
            draw_body: Whether to draw body skeleton
            draw_hands: Whether to draw hand keypoints
            line_thickness: Line thickness for skeleton
            point_radius: Radius for keypoint circles

        Returns:
            Image with rendered pose
        """
        output = image.copy()

        # Body skeleton connections (COCO format)
        body_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
            (1, 5), (5, 6), (6, 7),  # Left arm
            (1, 8), (8, 9), (9, 10),  # Right leg
            (1, 11), (11, 12), (12, 13),  # Left leg
            (0, 14), (14, 16),  # Right face
            (0, 15), (15, 17),  # Left face
        ]

        if draw_body and pose.body is not None:
            # Draw connections
            for start, end in body_connections:
                if start < len(pose.body) and end < len(pose.body):
                    pt1 = pose.body[start]
                    pt2 = pose.body[end]
                    if pt1[2] > 0.3 and pt2[2] > 0.3:
                        cv2.line(
                            output,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            (0, 255, 0),
                            line_thickness,
                        )

            # Draw keypoints
            for kpt in pose.body:
                if kpt[2] > 0.3:
                    cv2.circle(
                        output,
                        (int(kpt[0]), int(kpt[1])),
                        point_radius,
                        (255, 0, 0),
                        -1,
                    )

        if draw_hands:
            for hand in [pose.left_hand, pose.right_hand]:
                if hand is not None:
                    # Draw hand connections
                    hand_connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    ]
                    for start, end in hand_connections:
                        pt1 = hand[start]
                        pt2 = hand[end]
                        cv2.line(
                            output,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            (0, 0, 255),
                            1,
                        )

        return output

    async def render_pose_video(
        self,
        sequence: PoseSequence,
        output_path: Path,
        background: str = "black",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Path:
        """
        Render pose sequence to a video file.

        Args:
            sequence: Pose sequence to render
            output_path: Output video path
            background: Background color or "transparent"
            width: Output width (default: sequence width)
            height: Output height (default: sequence height)

        Returns:
            Path to rendered video
        """
        w = width or sequence.width or 512
        h = height or sequence.height or 512

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, sequence.fps, (w, h))

        try:
            for frame_poses in sequence.frames:
                # Create background
                if background == "black":
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                elif background == "white":
                    frame = np.ones((h, w, 3), dtype=np.uint8) * 255
                else:
                    frame = np.zeros((h, w, 3), dtype=np.uint8)

                # Render all poses in frame
                for pose in frame_poses:
                    frame = self.render_pose(frame, pose)

                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            logger.info(f"Rendered pose video: {output_path}")
            return output_path

        finally:
            out.release()

    def poses_to_controlnet_format(
        self,
        sequence: PoseSequence,
    ) -> List[np.ndarray]:
        """
        Convert pose sequence to ControlNet-compatible format.

        Returns:
            List of pose images for each frame
        """
        frames = []
        for frame_poses in sequence.frames:
            # Create black background
            frame = np.zeros((sequence.height, sequence.width, 3), dtype=np.uint8)

            for pose in frame_poses:
                frame = self.render_pose(frame, pose)

            frames.append(frame)

        return frames

    def close(self) -> None:
        """Release resources."""
        if self._model is not None:
            if hasattr(self._model, "close"):
                self._model.close()
            self._model = None

        if hasattr(self, "_hand_model") and self._hand_model is not None:
            if hasattr(self._hand_model, "close"):
                self._hand_model.close()
            self._hand_model = None

        self._initialized = False


# Singleton instance
_extractor: Optional[PoseExtractor] = None


def get_pose_extractor(backend: str = "mediapipe") -> PoseExtractor:
    """Get the global pose extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = PoseExtractor(backend=backend)
    return _extractor
