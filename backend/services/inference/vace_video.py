"""
VACE-style offline video rendering service.

This service provides deterministic built-in renderers for pose and motion
transfer so the advertised offline modes always produce valid output videos,
even when external Wan/VACE model integrations are not available.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

import cv2
import numpy as np
from loguru import logger

from backend.core.config import settings
from backend.core.models import GenerateRequest, QualityPreset
from backend.core.exceptions import InvalidImageError, InvalidVideoError, VideoProcessingError
from backend.services.pose_extractor import PoseKeypoints, get_pose_extractor


ProgressCallback = Callable[[float, str], Awaitable[None] | None]


class VaceVideoService:
    """Render pose- and motion-transfer videos from a reference image and source video."""

    def __init__(self) -> None:
        self.pose_extractor = get_pose_extractor()
        self._fallback_warning_logged = False

    async def generate_pose_transfer(
        self,
        reference_image_path: Path,
        input_video_path: Path,
        output_path: Path,
        request: GenerateRequest,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Render a character overlay that follows the subject pose in the input video."""
        return await self._generate_transfer(
            mode="pose",
            reference_image_path=reference_image_path,
            input_video_path=input_video_path,
            output_path=output_path,
            request=request,
            progress_callback=progress_callback,
        )

    async def generate_motion_transfer(
        self,
        reference_image_path: Path,
        input_video_path: Path,
        output_path: Path,
        request: GenerateRequest,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Render a stylized motion transfer video driven by the input poses."""
        return await self._generate_transfer(
            mode="motion",
            reference_image_path=reference_image_path,
            input_video_path=input_video_path,
            output_path=output_path,
            request=request,
            progress_callback=progress_callback,
        )

    async def _generate_transfer(
        self,
        *,
        mode: str,
        reference_image_path: Path,
        input_video_path: Path,
        output_path: Path,
        request: GenerateRequest,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        reference_rgba = self._load_reference_image(reference_image_path)
        capture = cv2.VideoCapture(str(input_video_path))
        if not capture.isOpened():
            raise InvalidVideoError(f"Could not open input video: {input_video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or float(settings.default_fps)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or settings.default_resolution[0])
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or settings.default_resolution[1])
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_duration = source_frames / source_fps if source_frames > 0 else float(request.duration or 0)

        target_fps = float(request.fps or source_fps or settings.default_fps)
        preview_mode = bool(request.extra_params.get("preview_mode"))
        max_duration = 2.0 if preview_mode else (request.duration or source_duration or 4.0)
        max_duration = max(1.0, max_duration)
        expected_frames = max(1, int(round(max_duration * target_fps)))
        frame_skip = max(1, int(round(source_fps / target_fps))) if source_fps > target_fps else 1
        if request.resolution:
            target_resolution = tuple(request.resolution)
        else:
            quality_scale = {
                QualityPreset.DRAFT: 0.75,
                QualityPreset.STANDARD: 1.0,
                QualityPreset.HIGH: 1.1,
                QualityPreset.ULTRA: 1.2,
            }.get(request.quality, 1.0)
            target_resolution = (
                int(source_width * quality_scale),
                int(source_height * quality_scale),
            )
        target_resolution = (
            max(2, int(target_resolution[0] // 2 * 2)),
            max(2, int(target_resolution[1] // 2 * 2)),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            target_fps,
            target_resolution,
        )
        if not writer.isOpened():
            capture.release()
            raise VideoProcessingError(f"Could not open output writer for {output_path}")

        seed = request.seed if request.seed is not None else abs(hash((request.prompt, mode))) % (2**32)
        rng = np.random.default_rng(seed)
        pose_backend = "builtin"
        processed_frames = 0
        read_frames = 0

        try:
            while processed_frames < expected_frames:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if read_frames % frame_skip != 0:
                    read_frames += 1
                    continue

                read_frames += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[1] != target_resolution[0] or frame_rgb.shape[0] != target_resolution[1]:
                    frame_rgb = cv2.resize(frame_rgb, target_resolution, interpolation=cv2.INTER_LINEAR)

                pose, pose_backend = await self._extract_primary_pose(frame_rgb)
                rendered = self._render_frame(
                    mode=mode,
                    frame_rgb=frame_rgb,
                    reference_rgba=reference_rgba,
                    pose=pose,
                    strength=request.strength,
                    preserve_background=request.preserve_background,
                    frame_index=processed_frames,
                    rng=rng,
                )
                writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

                processed_frames += 1
                if progress_callback and (processed_frames == 1 or processed_frames % 5 == 0 or processed_frames == expected_frames):
                    await self._emit_progress(
                        progress_callback,
                        processed_frames / expected_frames,
                        f"Rendering {mode} transfer frame {processed_frames}/{expected_frames}",
                    )
        finally:
            capture.release()
            writer.release()

        if processed_frames == 0:
            raise VideoProcessingError("No frames were rendered for the VACE transfer output")

        logger.info(
            f"Rendered {mode} transfer video with {processed_frames} frames "
            f"using {pose_backend} pose backend"
        )
        return {
            "output_path": output_path,
            "metadata": {
                "implementation": "builtin_renderer",
                "pose_backend": pose_backend,
                "frames_rendered": processed_frames,
                "fps": target_fps,
                "resolution": target_resolution,
            },
        }

    def _load_reference_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise InvalidImageError(f"Could not load reference image: {image_path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return image

    async def _extract_primary_pose(self, frame_rgb: np.ndarray) -> tuple[PoseKeypoints, str]:
        try:
            poses = await self.pose_extractor.extract_from_image(
                frame_rgb,
                detect_hands=False,
                detect_face=False,
            )
            if poses:
                primary = max(poses, key=lambda pose: pose.confidence)
                return primary, self.pose_extractor.backend
        except Exception as exc:
            if not self._fallback_warning_logged:
                logger.warning(f"Pose extraction unavailable, using fallback pose heuristic: {exc}")
                self._fallback_warning_logged = True
        return self._build_fallback_pose(frame_rgb.shape[1], frame_rgb.shape[0]), "builtin"

    def _build_fallback_pose(self, width: int, height: int) -> PoseKeypoints:
        cx = width / 2.0
        cy = height / 2.0
        scale = min(width, height) * 0.18
        body = np.array(
            [
                [cx, cy - scale * 1.8, 0.95],  # nose
                [cx, cy - scale * 1.1, 0.95],  # neck
                [cx + scale * 0.9, cy - scale * 0.9, 0.9],
                [cx + scale * 1.2, cy - scale * 0.1, 0.9],
                [cx + scale * 1.3, cy + scale * 0.8, 0.85],
                [cx - scale * 0.9, cy - scale * 0.9, 0.9],
                [cx - scale * 1.2, cy - scale * 0.1, 0.9],
                [cx - scale * 1.3, cy + scale * 0.8, 0.85],
                [cx + scale * 0.6, cy + scale * 0.5, 0.9],
                [cx + scale * 0.65, cy + scale * 1.8, 0.85],
                [cx + scale * 0.7, cy + scale * 3.0, 0.8],
                [cx - scale * 0.6, cy + scale * 0.5, 0.9],
                [cx - scale * 0.65, cy + scale * 1.8, 0.85],
                [cx - scale * 0.7, cy + scale * 3.0, 0.8],
                [cx + scale * 0.2, cy - scale * 2.0, 0.8],
                [cx - scale * 0.2, cy - scale * 2.0, 0.8],
                [cx + scale * 0.5, cy - scale * 1.9, 0.8],
                [cx - scale * 0.5, cy - scale * 1.9, 0.8],
            ],
            dtype=np.float32,
        )
        bbox = (
            max(0.0, cx - scale * 1.8),
            max(0.0, cy - scale * 2.2),
            min(float(width), cx + scale * 1.8),
            min(float(height), cy + scale * 3.3),
        )
        return PoseKeypoints(body=body, bbox=bbox, confidence=0.75)

    def _render_frame(
        self,
        *,
        mode: str,
        frame_rgb: np.ndarray,
        reference_rgba: np.ndarray,
        pose: PoseKeypoints,
        strength: float,
        preserve_background: bool,
        frame_index: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        bbox = self._pose_bbox(pose, frame_rgb.shape)
        if mode == "pose":
            background = frame_rgb.copy() if preserve_background else self._soft_background(frame_rgb)
            character = self._prepare_character_overlay(reference_rgba, bbox, strength)
            return self._alpha_blend(background, character, bbox)

        background = self._motion_background(frame_rgb, reference_rgba, preserve_background, frame_index, rng)
        try:
            background = self.pose_extractor.render_pose(
                background,
                pose,
                draw_body=True,
                draw_hands=False,
                line_thickness=2,
                point_radius=3,
            )
        except Exception:
            pass
        character = self._prepare_character_overlay(reference_rgba, bbox, min(1.0, strength + 0.05))
        composed = self._alpha_blend(background, character, bbox)
        if preserve_background:
            return cv2.addWeighted(composed, 0.78, frame_rgb, 0.22, 0.0)
        return composed

    def _pose_bbox(self, pose: PoseKeypoints, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        if pose.bbox is not None:
            x1, y1, x2, y2 = pose.bbox
        else:
            valid = pose.body[pose.body[:, 2] > 0.3]
            if len(valid) == 0:
                return int(width * 0.25), int(height * 0.1), int(width * 0.75), int(height * 0.9)
            x1, y1 = valid[:, :2].min(axis=0)
            x2, y2 = valid[:, :2].max(axis=0)
        pad_x = (x2 - x1) * 0.18
        pad_y = (y2 - y1) * 0.12
        return (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)),
            min(height, int(y2 + pad_y)),
        )

    def _prepare_character_overlay(
        self,
        reference_rgba: np.ndarray,
        bbox: tuple[int, int, int, int],
        strength: float,
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        target_w = max(2, x2 - x1)
        target_h = max(2, y2 - y1)
        ref_h, ref_w = reference_rgba.shape[:2]
        scale = min(target_w / ref_w, target_h / ref_h)
        scaled_w = max(2, int(ref_w * scale))
        scaled_h = max(2, int(ref_h * scale))
        resized = cv2.resize(reference_rgba, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        offset_x = (target_w - scaled_w) // 2
        offset_y = (target_h - scaled_h) // 2
        canvas[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w] = resized
        yy, xx = np.ogrid[:target_h, :target_w]
        center_x = target_w / 2.0
        center_y = target_h / 2.0
        radius_x = max(1.0, target_w * 0.48)
        radius_y = max(1.0, target_h * 0.5)
        ellipse = ((xx - center_x) ** 2) / (radius_x ** 2) + ((yy - center_y) ** 2) / (radius_y ** 2)
        feather = np.clip(1.0 - (ellipse - 0.78) / 0.22, 0.0, 1.0)
        alpha = canvas[:, :, 3].astype(np.float32) / 255.0
        alpha = np.clip(alpha * feather * np.clip(strength, 0.15, 1.0), 0.0, 1.0)
        canvas[:, :, 3] = np.uint8(alpha * 255.0)
        return canvas

    def _soft_background(self, frame_rgb: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame_rgb, (0, 0), 11)
        return cv2.addWeighted(blurred, 0.75, np.full_like(frame_rgb, 12), 0.25, 0.0)

    def _motion_background(
        self,
        frame_rgb: np.ndarray,
        reference_rgba: np.ndarray,
        preserve_background: bool,
        frame_index: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if preserve_background:
            base = cv2.GaussianBlur(frame_rgb, (0, 0), 15)
            return cv2.addWeighted(base, 0.6, np.full_like(base, 24), 0.4, 0.0)

        ref_rgb = reference_rgba[:, :, :3]
        height, width = frame_rgb.shape[:2]
        scale = max(width / ref_rgb.shape[1], height / ref_rgb.shape[0])
        resized = cv2.resize(
            ref_rgb,
            (max(2, int(ref_rgb.shape[1] * scale)), max(2, int(ref_rgb.shape[0] * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        phase = frame_index * 0.8
        max_x = max(0, resized.shape[1] - width)
        max_y = max(0, resized.shape[0] - height)
        start_x = int((math.sin(phase * 0.07) * 0.5 + 0.5) * max_x) if max_x else 0
        start_y = int((math.cos(phase * 0.05) * 0.5 + 0.5) * max_y) if max_y else 0
        crop = resized[start_y:start_y + height, start_x:start_x + width]
        crop = cv2.GaussianBlur(crop, (0, 0), 19)
        noise = rng.normal(0.0, 4.0, crop.shape).astype(np.float32)
        return np.clip(crop.astype(np.float32) * 0.72 + noise + 18.0, 0, 255).astype(np.uint8)

    def _alpha_blend(
        self,
        background_rgb: np.ndarray,
        overlay_rgba: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        output = background_rgb.copy()
        overlay_h = min(overlay_rgba.shape[0], output.shape[0] - y1, y2 - y1)
        overlay_w = min(overlay_rgba.shape[1], output.shape[1] - x1, x2 - x1)
        if overlay_h <= 0 or overlay_w <= 0:
            return output
        overlay = overlay_rgba[:overlay_h, :overlay_w]
        target = output[y1:y1 + overlay_h, x1:x1 + overlay_w]
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        blended = overlay[:, :, :3].astype(np.float32) * alpha + target.astype(np.float32) * (1.0 - alpha)
        output[y1:y1 + overlay_h, x1:x1 + overlay_w] = np.clip(blended, 0, 255).astype(np.uint8)
        return output

    async def _emit_progress(self, callback: ProgressCallback, ratio: float, step: str) -> None:
        result = callback(max(0.0, min(1.0, ratio)), step)
        if asyncio.iscoroutine(result):
            await result


_vace_service: Optional[VaceVideoService] = None


def get_vace_video_service() -> VaceVideoService:
    global _vace_service
    if _vace_service is None:
        _vace_service = VaceVideoService()
    return _vace_service
