"""
Face Swapping Inference Service.
Replaces ComfyUI's ReActor node.
"""

import os
import cv2
import numpy as np
import insightface
from loguru import logger
from backend.core.config import settings
from backend.services.face_detector import DetectedFace, get_face_detector
from backend.services.model_manager import get_model_manager

class FaceSwapper:
    def __init__(self):
        self.face_detector = get_face_detector()
        self.model_manager = get_model_manager()
        self.device = settings.device
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Cache for source face to avoid redundant detection in real-time processing
        self._cached_source_face = None
        self._cached_source_hash = None
        self._realtime_detection_max_size = 512

    def _select_detection_max_size(self, target_img: np.ndarray, realtime_config: dict | None) -> int:
        """Choose a bounded high-res analysis size for target face detection."""
        if not realtime_config:
            return self._realtime_detection_max_size

        output_resolution = realtime_config.get("output_resolution") or (
            target_img.shape[1],
            target_img.shape[0],
        )
        max_output_dim = max(output_resolution)

        if realtime_config.get("full_frame_inference", True):
            if max_output_dim >= 3840:
                return 1920
            if max_output_dim >= 2560:
                return 1536
            if max_output_dim >= 1920:
                return 1280
        return self._realtime_detection_max_size
        
    def _load_swapper(self):
        """Load the inswapper model via ModelManager."""
        def loader():
            model_path = settings.models_dir / "insightface" / "inswapper_128.onnx"
            if not model_path.exists():
                raise FileNotFoundError(f"Face swap model not found at {model_path}")
            
            providers = []
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif self.device == "mps":
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                
            return insightface.model_zoo.get_model(str(model_path), providers=providers)
            
        return self.model_manager.load_model("inswapper_128", loader)

    def _compute_image_hash(self, image: np.ndarray) -> int:
        """Compute a hash for an image to detect changes."""
        # Use a simple hash based on image shape and sampled pixels for speed
        # This is faster than full image hashing while still detecting most changes
        h = hash((image.shape, image.dtype.name))
        # Sample corners and center for change detection
        if image.size > 0:
            h ^= hash(image[0, 0].tobytes())
            h ^= hash(image[-1, -1].tobytes())
            h ^= hash(image[image.shape[0]//2, image.shape[1]//2].tobytes())
        return h

    async def set_source_face(self, source_img: np.ndarray) -> bool:
        """
        Pre-extract and cache the source face for efficient real-time processing.
        
        Call this once with the source image before processing video frames.
        Returns True if a face was successfully detected and cached.
        """
        try:
            source_face = await self.face_detector.get_primary_face(source_img)
            self._cached_source_face = source_face
            self._cached_source_hash = self._compute_image_hash(source_img)
            logger.info("Source face cached for real-time processing")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache source face: {e}")
            self._cached_source_face = None
            self._cached_source_hash = None
            return False

    def clear_source_cache(self) -> None:
        """Clear the cached source face."""
        self._cached_source_face = None
        self._cached_source_hash = None

    async def _get_source_face(self, source_img: np.ndarray):
        """
        Get source face, using cache if available and image hasn't changed.
        """
        current_hash = self._compute_image_hash(source_img)
        
        # Use cached face if available and source image hasn't changed
        if self._cached_source_face is not None and self._cached_source_hash == current_hash:
            return self._cached_source_face
        
        # Detect and cache the new source face
        source_face = await self.face_detector.get_primary_face(source_img)
        self._cached_source_face = source_face
        self._cached_source_hash = current_hash
        return source_face

    def _resize_for_realtime_detection(
        self,
        image: np.ndarray,
        max_detection_size: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Downscale target frames before detection to keep realtime latency low."""
        height, width = image.shape[:2]
        max_dim = max(height, width)
        detection_limit = max_detection_size or self._realtime_detection_max_size
        if max_dim <= detection_limit:
            return image, 1.0

        scale = detection_limit / float(max_dim)
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _restore_detected_face_scale(self, face: DetectedFace, scale: float) -> DetectedFace:
        """Map face detection results from the resized frame back to the source frame."""
        if scale == 1.0:
            return face

        inverse_scale = 1.0 / scale
        bbox = tuple(float(coord * inverse_scale) for coord in face.bbox)
        landmarks = face.landmarks.astype(np.float32) * inverse_scale
        return DetectedFace(
            bbox=bbox,
            landmarks=landmarks,
            confidence=face.confidence,
            embedding=face.embedding,
            age=face.age,
            gender=face.gender,
            aligned_face=face.aligned_face,
        )

    def _apply_highres_face_detail(
        self,
        image: np.ndarray,
        target_face: DetectedFace,
        realtime_config: dict | None,
    ) -> np.ndarray:
        """Enhance the swapped face ROI when preserving a high-resolution output."""
        if not realtime_config or not realtime_config.get("full_frame_inference", True):
            return image

        output_resolution = realtime_config.get("output_resolution") or (image.shape[1], image.shape[0])
        if max(output_resolution) < 1920:
            return image

        x1, y1, x2, y2 = [int(v) for v in target_face.bbox]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = max(8, int(width * 0.15))
        pad_y = max(8, int(height * 0.15))
        roi_x1 = max(0, x1 - pad_x)
        roi_y1 = max(0, y1 - pad_y)
        roi_x2 = min(image.shape[1], x2 + pad_x)
        roi_y2 = min(image.shape[0], y2 + pad_y)

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return image

        blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(roi, 1.18, blurred, -0.18, 0)

        output = image.copy()
        output[roi_y1:roi_y2, roi_x1:roi_x2] = sharpened
        return output

    async def swap_face(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        realtime_config: dict | None = None,
    ) -> np.ndarray:
        """
        Swap face from source_img into target_img.
        """
        swapper = self._load_swapper()
        
        # 1. Get source face (uses cache for real-time efficiency)
        try:
            source_face = await self._get_source_face(source_img)
        except Exception as e:
            logger.warning(f"Could not find source face: {e}")
            return target_img

        # 2. Detect the primary target face using a high-res aware analysis image.
        detection_image, detection_scale = self._resize_for_realtime_detection(
            target_img,
            max_detection_size=self._select_detection_max_size(target_img, realtime_config),
        )
        target_faces = await self.face_detector.detect_faces(
            detection_image,
            max_faces=1,
            extract_embedding=False,
        )
        if not target_faces:
            return target_img
            
        # 3. Perform swap
        result_img = target_img.copy()
        
        # In a real app, you might select which target face to swap
        # For now, swap the largest one (primary)
        target_face = self._restore_detected_face_scale(target_faces[0], detection_scale)
        
        # InsightFace's swapper expects raw face objects, but our detector wraps them.
        # However, the swapper mainly needs the kps (landmarks) and the source embedding.
        # We need to construct a simplified object that the swapper accepts if we can't get the raw one easily.
        # Fortunately, the inswapper usually takes the target image, target face object (with kps), 
        # and source face object (with embedding) and paste_back=True.
        
        # Since our DetectedFace is a wrapper, we might need to adapt.
        # Let's assume for this implementation we can construct a dummy object 
        # or that we modify FaceDetector to return raw objects if needed.
        # For this prototype, I'll rely on the swapper's expected interface.
        
        # Construct dummy target face for inswapper
        class Face:
            def __init__(self, kps):
                self.kps = kps
        
        tf_obj = Face(target_face.landmarks)
        
        # Construct dummy source face
        sf_obj = Face(None)
        sf_obj.embedding = source_face.embedding
        sf_obj.normed_embedding = source_face.embedding / np.linalg.norm(source_face.embedding)
        
        result_img = swapper.get(result_img, tf_obj, sf_obj, paste_back=True)
        result_img = self._apply_highres_face_detail(result_img, target_face, realtime_config)
        
        return result_img

_face_swapper = None

def get_face_swapper() -> FaceSwapper:
    global _face_swapper
    if _face_swapper is None:
        _face_swapper = FaceSwapper()
    return _face_swapper
