"""
LivePortrait Inference Service.
"""

import asyncio
from dataclasses import dataclass
import numpy as np
from typing import Optional, Any
import cv2
from loguru import logger
from backend.core.config import settings
from backend.services.face_detector import get_face_detector
from backend.services.model_manager import get_model_manager


@dataclass
class PortraitState:
    """Detected portrait state used for landmark-driven animation."""

    landmarks: np.ndarray
    bbox: tuple[int, int, int, int]
    method: str


class LivePortraitService:
    def __init__(self):
        self.model_manager = get_model_manager()
        self.face_detector = get_face_detector()
        self.device = settings.device
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self._face_mesh = None
        self._face_mesh_static = None
        self._source_state: Optional[PortraitState] = None
        self._source_hash: Optional[int] = None
        self._dense_landmark_count = 96

    def _compute_image_hash(self, image: np.ndarray) -> int:
        h = hash((image.shape, image.dtype.name))
        if image.size > 0:
            h ^= hash(image[0, 0].tobytes())
            h ^= hash(image[-1, -1].tobytes())
            h ^= hash(image[image.shape[0] // 2, image.shape[1] // 2].tobytes())
        return h

    def _load_pipeline(self):
        """Load optional LivePortrait pipeline if present."""
        def loader():
            try:
                from liveportrait import LivePortraitPipeline
            except ImportError:
                logger.info("External LivePortrait package not found, using built-in landmark renderer")
                return None

            try:
                return LivePortraitPipeline(
                    inference_cfg={
                        "device_id": 0 if self.device == "cuda" else "cpu",
                        "flag_use_half_precision": settings.enable_fp16 and self.device == "cuda",
                    }
                )
            except Exception as exc:  # pragma: no cover - depends on optional package
                logger.warning(f"Failed to initialize external LivePortrait pipeline: {exc}")
                return None

        return self.model_manager.load_model("live_portrait", loader)

    def _ensure_face_mesh(self) -> bool:
        """Initialize MediaPipe FaceMesh if available."""
        if self._face_mesh is not None and self._face_mesh_static is not None:
            return True

        try:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_mesh_static = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return True
        except ImportError:
            logger.warning("MediaPipe not available, LivePortrait will use sparse-landmark fallback")
        except Exception as exc:
            logger.warning(f"Failed to initialize MediaPipe FaceMesh: {exc}")
        return False

    async def _detect_with_facemesh(
        self,
        image: np.ndarray,
        *,
        static_image: bool,
        resize_limit: int | None = None,
    ) -> Optional[PortraitState]:
        if not self._ensure_face_mesh():
            return None

        height, width = image.shape[:2]
        scaled_image = image
        scale = 1.0
        if resize_limit and max(height, width) > resize_limit:
            scale = resize_limit / float(max(height, width))
            scaled_image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        mesh = self._face_mesh_static if static_image else self._face_mesh
        results = await asyncio.to_thread(mesh.process, scaled_image)
        if not results.multi_face_landmarks:
            return None

        mesh_landmarks = results.multi_face_landmarks[0].landmark
        scaled_height, scaled_width = scaled_image.shape[:2]
        points = np.array(
            [[landmark.x * scaled_width, landmark.y * scaled_height] for landmark in mesh_landmarks],
            dtype=np.float32,
        )
        if scale != 1.0:
            points /= scale

        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        bbox = (
            max(0, int(x1)),
            max(0, int(y1)),
            min(width, int(x2)),
            min(height, int(y2)),
        )
        return PortraitState(landmarks=points, bbox=bbox, method="facemesh")

    async def _detect_with_face_detector(self, image: np.ndarray) -> Optional[PortraitState]:
        try:
            face = await self.face_detector.get_primary_face(image)
        except Exception:
            return None

        landmarks = np.asarray(face.landmarks, dtype=np.float32)
        bbox = tuple(int(value) for value in face.bbox)
        return PortraitState(landmarks=landmarks, bbox=bbox, method="insightface")

    async def _detect_portrait_state(
        self,
        image: np.ndarray,
        *,
        static_image: bool,
        realtime_config: Optional[dict] = None,
    ) -> Optional[PortraitState]:
        resize_limit = None
        if realtime_config and not realtime_config.get("full_frame_inference", True):
            resize_limit = 1280

        state = await self._detect_with_facemesh(
            image,
            static_image=static_image,
            resize_limit=resize_limit,
        )
        if state is not None:
            return state
        return await self._detect_with_face_detector(image)

    def _select_landmark_subset(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = min(len(source_points), len(target_points))
        if count < 3:
            raise ValueError("Not enough landmarks for portrait animation")

        if count > self._dense_landmark_count:
            indices = np.linspace(0, count - 1, num=self._dense_landmark_count, dtype=int)
            indices = np.unique(indices)
            source_points = source_points[indices]
            target_points = target_points[indices]
        else:
            source_points = source_points[:count]
            target_points = target_points[:count]

        return source_points.astype(np.float32), target_points.astype(np.float32)

    def _crop_roi(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        padding_ratio: float = 0.2,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = int(width * padding_ratio)
        pad_y = int(height * padding_ratio)
        roi_x1 = max(0, x1 - pad_x)
        roi_y1 = max(0, y1 - pad_y)
        roi_x2 = min(image.shape[1], x2 + pad_x)
        roi_y2 = min(image.shape[0], y2 + pad_y)
        return image[roi_y1:roi_y2, roi_x1:roi_x2].copy(), (roi_x1, roi_y1, roi_x2, roi_y2)

    def _boundary_points(self, width: int, height: int) -> np.ndarray:
        return np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
                [width * 0.5, 0],
                [width - 1, height * 0.5],
                [width * 0.5, height - 1],
                [0, height * 0.5],
            ],
            dtype=np.float32,
        )

    def _piecewise_affine_warp(
        self,
        source_roi: np.ndarray,
        source_points: np.ndarray,
        target_shape: tuple[int, int, int],
        target_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        from scipy.spatial import Delaunay

        height, width = target_shape[:2]
        source_points = np.vstack([source_points, self._boundary_points(source_roi.shape[1], source_roi.shape[0])])
        target_points = np.vstack([target_points, self._boundary_points(width, height)])
        delaunay = Delaunay(target_points)

        output = np.zeros((height, width, 3), dtype=np.float32)
        weights = np.zeros((height, width, 1), dtype=np.float32)

        for simplex in delaunay.simplices:
            src_tri = source_points[simplex].astype(np.float32)
            dst_tri = target_points[simplex].astype(np.float32)

            src_rect = cv2.boundingRect(src_tri)
            dst_rect = cv2.boundingRect(dst_tri)
            if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
                continue

            src_x, src_y, src_w, src_h = src_rect
            dst_x, dst_y, dst_w, dst_h = dst_rect

            src_patch = source_roi[src_y:src_y + src_h, src_x:src_x + src_w]
            if src_patch.size == 0:
                continue

            src_tri_local = src_tri - np.array([src_x, src_y], dtype=np.float32)
            dst_tri_local = dst_tri - np.array([dst_x, dst_y], dtype=np.float32)
            warp_matrix = cv2.getAffineTransform(src_tri_local[:3], dst_tri_local[:3])
            warped_patch = cv2.warpAffine(
                src_patch,
                warp_matrix,
                (dst_w, dst_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            triangle_mask = np.zeros((dst_h, dst_w, 1), dtype=np.float32)
            cv2.fillConvexPoly(
                triangle_mask,
                np.int32(dst_tri_local),
                (1.0,),
                lineType=cv2.LINE_AA,
            )

            output_slice = output[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w]
            weight_slice = weights[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w]
            output_slice += warped_patch.astype(np.float32) * triangle_mask
            weight_slice += triangle_mask

        warped = output / np.clip(weights, 1e-6, None)
        valid_mask = np.clip(weights, 0.0, 1.0)
        return warped.astype(np.uint8), valid_mask

    def _blend_face(
        self,
        driving_frame: np.ndarray,
        roi_rect: tuple[int, int, int, int],
        warped_face: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        x1, y1, x2, y2 = roi_rect
        output = driving_frame.copy()
        target_roi = output[y1:y2, x1:x2]
        if target_roi.shape[:2] != warped_face.shape[:2]:
            return output

        if mask.ndim == 2:
            mask = mask[..., None]
        blur_size = max(3, int(min(mask.shape[:2]) * 0.08) | 1)
        smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
        smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
        blended = (
            warped_face.astype(np.float32) * smooth_mask
            + target_roi.astype(np.float32) * (1.0 - smooth_mask)
        )
        output[y1:y2, x1:x2] = blended.astype(np.uint8)
        return output

    async def _get_source_state(self, source_img: np.ndarray) -> PortraitState:
        source_hash = self._compute_image_hash(source_img)
        if self._source_state is not None and self._source_hash == source_hash:
            return self._source_state

        source_state = await self._detect_portrait_state(source_img, static_image=True)
        if source_state is None:
            raise ValueError("Could not detect source portrait landmarks")

        self._source_state = source_state
        self._source_hash = source_hash
        return source_state

    def process(self, source_img: np.ndarray, driving_video_path: str) -> str:
        """
        Animate source image using driving video.
        Returns path to output video.
        """
        self._load_pipeline()
        return driving_video_path

    async def process_frame(
        self,
        source_img: np.ndarray,
        driving_frame: np.ndarray,
        realtime_config: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Animate source image using a single driving frame.
        """
        self._load_pipeline()

        try:
            source_state = await self._get_source_state(source_img)
            driving_state = await self._detect_portrait_state(
                driving_frame,
                static_image=False,
                realtime_config=realtime_config,
            )
            if driving_state is None:
                return driving_frame

            source_points, target_points = self._select_landmark_subset(
                source_state.landmarks,
                driving_state.landmarks,
            )
            source_roi, source_rect = self._crop_roi(source_img, source_state.bbox)
            _, target_rect = self._crop_roi(driving_frame, driving_state.bbox)

            src_x1, src_y1, _, _ = source_rect
            dst_x1, dst_y1, dst_x2, dst_y2 = target_rect
            source_points_local = source_points - np.array([src_x1, src_y1], dtype=np.float32)
            target_points_local = target_points - np.array([dst_x1, dst_y1], dtype=np.float32)

            warped_face, mask = self._piecewise_affine_warp(
                source_roi,
                source_points_local,
                (dst_y2 - dst_y1, dst_x2 - dst_x1, 3),
                target_points_local,
            )
            return self._blend_face(driving_frame, target_rect, warped_face, mask)
        except Exception as exc:
            logger.warning(f"Built-in LivePortrait frame rendering failed: {exc}")
            return driving_frame

    def close(self) -> None:
        """Release model and detector resources."""
        for mesh in (self._face_mesh, self._face_mesh_static):
            if mesh is not None:
                try:
                    mesh.close()
                except Exception:
                    pass
        self._face_mesh = None
        self._face_mesh_static = None
        self._source_state = None
        self._source_hash = None


class MockLivePortraitPipeline:
    def __init__(self, *args, **kwargs):
        pass
        
    def execute(self, *args, **kwargs):
        logger.info("Mock LivePortrait execution")
        return None

_live_portrait = None

def get_live_portrait_service() -> LivePortraitService:
    global _live_portrait
    if _live_portrait is None:
        _live_portrait = LivePortraitService()
    return _live_portrait
