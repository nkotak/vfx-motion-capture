"""
Wan-style reference-to-video generation service.

The service prefers a configured external diffusion pipeline when available,
but always falls back to a deterministic built-in renderer that consumes the
reference image, prompt, and generation settings to produce a valid video.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from loguru import logger

from backend.core.config import settings
from backend.core.models import GenerateRequest, QualityPreset
from backend.core.exceptions import InvalidImageError, VideoProcessingError
from backend.services.model_manager import get_model_manager

try:  # pragma: no cover - optional heavy dependency
    import torch
except ImportError:  # pragma: no cover - handled at runtime
    torch = None


class WanVideoService:
    def __init__(self) -> None:
        self.model_manager = get_model_manager()
        self.device = self._resolve_device()

    def _resolve_device(self) -> str:
        if settings.device != "auto":
            return settings.device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _model_source(self) -> Optional[str]:
        return settings.wan_r2v_model_path

    def _load_pipeline(self):
        model_source = self._model_source()
        if not model_source:
            return None

        def loader():
            try:
                from diffusers import DiffusionPipeline
            except ImportError:
                logger.warning("diffusers is not installed; using built-in Wan renderer")
                return None

            if torch is None:
                logger.warning("torch is not installed; using built-in Wan renderer")
                return None

            try:
                pipe = DiffusionPipeline.from_pretrained(
                    model_source,
                    torch_dtype=torch.float16 if settings.enable_fp16 and self.device == "cuda" else torch.float32,
                )
                if hasattr(pipe, "to"):
                    pipe.to(self.device)
                return pipe
            except Exception as exc:  # pragma: no cover - depends on external models
                logger.warning(f"Failed to load Wan pipeline '{model_source}': {exc}")
                return None

        return self.model_manager.load_model(f"wan_video::{model_source}", loader)

    async def generate(
        self,
        prompt: str,
        reference_image: np.ndarray,
        output_path: Path,
        request: GenerateRequest,
    ) -> Dict[str, Any]:
        """Generate a reference-conditioned video and return output metadata."""
        if reference_image is None:
            raise InvalidImageError("Reference image is required for Wan generation")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pipe = self._load_pipeline()
        if pipe is not None:
            try:
                generated = await self._generate_with_pipeline(pipe, prompt, reference_image, output_path, request)
                if generated is not None:
                    return generated
            except Exception as exc:  # pragma: no cover - external models are optional
                logger.warning(f"External Wan pipeline failed, using built-in renderer: {exc}")

        return await self._generate_with_builtin(prompt, reference_image, output_path, request)

    async def _generate_with_pipeline(
        self,
        pipe: Any,
        prompt: str,
        reference_image: np.ndarray,
        output_path: Path,
        request: GenerateRequest,
    ) -> Optional[Dict[str, Any]]:
        """Best-effort external pipeline integration when a compatible model exists."""
        if torch is None:  # pragma: no cover - protected by loader
            return None

        frame_count = self._frame_count(request)
        width, height = self._target_resolution(reference_image, request.resolution, request.quality)
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device if self.device != "mps" else "cpu").manual_seed(request.seed)

        def _invoke_pipeline():
            kwargs = {
                "prompt": prompt,
                "num_frames": frame_count,
                "width": width,
                "height": height,
                "generator": generator,
            }

            try:
                parameter_names = set(inspect.signature(pipe.__call__).parameters)
            except (TypeError, ValueError):
                parameter_names = set()

            if "image" in parameter_names:
                kwargs["image"] = self._to_bgr_pil(reference_image)
            elif "reference_image" in parameter_names:
                kwargs["reference_image"] = self._to_bgr_pil(reference_image)

            return pipe(**kwargs)

        result = await asyncio.to_thread(_invoke_pipeline)
        frames = getattr(result, "frames", None)
        if frames is None:
            frames = getattr(result, "images", None)
        if frames is None:
            frames = getattr(result, "videos", None)
        if frames is None or len(frames) == 0:
            return None

        first = frames[0]
        if isinstance(first, np.ndarray) and first.ndim == 4:
            frame_list = [frame for frame in first]
        elif isinstance(first, np.ndarray) and first.ndim == 3:
            frame_list = frames
        elif hasattr(first, "convert"):
            frame_list = [np.asarray(frame.convert("RGB")) for frame in frames]
        elif isinstance(first, (list, tuple)) and first and hasattr(first[0], "convert"):
            frame_list = [np.asarray(frame.convert("RGB")) for frame in first]
        else:
            frame_list = [np.asarray(frame) for frame in frames]

        self._write_video(frame_list, output_path, float(request.fps))
        return {
            "output_path": output_path,
            "metadata": {
                "implementation": "diffusion_pipeline",
                "model_source": self._model_source(),
                "frames_rendered": len(frame_list),
                "fps": float(request.fps),
                "resolution": (width, height),
            },
        }

    async def _generate_with_builtin(
        self,
        prompt: str,
        reference_image: np.ndarray,
        output_path: Path,
        request: GenerateRequest,
    ) -> Dict[str, Any]:
        reference_rgb = self._to_rgb(reference_image)
        width, height = self._target_resolution(reference_rgb, request.resolution, request.quality)
        frame_count = self._frame_count(request)
        seed = request.seed if request.seed is not None else self._prompt_seed(prompt)
        rng = np.random.default_rng(seed)
        motion = self._motion_profile(prompt, request.quality)

        frames = []
        for frame_index in range(frame_count):
            frames.append(
                self._render_builtin_frame(
                    reference_rgb=reference_rgb,
                    prompt=prompt,
                    frame_index=frame_index,
                    frame_count=frame_count,
                    width=width,
                    height=height,
                    rng=rng,
                    motion=motion,
                )
            )

        self._write_video(frames, output_path, float(request.fps))
        logger.info(f"Rendered Wan builtin video with {frame_count} frames for prompt '{prompt}'")
        return {
            "output_path": output_path,
            "metadata": {
                "implementation": "builtin_renderer",
                "model_source": self._model_source(),
                "frames_rendered": frame_count,
                "fps": float(request.fps),
                "resolution": (width, height),
                "seed": seed,
            },
        }

    def _render_builtin_frame(
        self,
        *,
        reference_rgb: np.ndarray,
        prompt: str,
        frame_index: int,
        frame_count: int,
        width: int,
        height: int,
        rng: np.random.Generator,
        motion: Dict[str, float],
    ) -> np.ndarray:
        background = self._cover_resize(reference_rgb, width, height)
        background = cv2.GaussianBlur(background, (0, 0), 21)
        background = self._apply_prompt_tint(background, prompt)

        progress = frame_index / max(1, frame_count - 1)
        zoom = 1.0 + motion["zoom"] * math.sin(progress * math.pi * motion["zoom_cycles"])
        pan_x = motion["pan_x"] * math.sin(progress * math.pi * 2.0)
        pan_y = motion["pan_y"] * math.cos(progress * math.pi * 1.5)
        rotation = motion["rotation"] * math.sin(progress * math.pi * 2.0)

        foreground = self._transform_reference(reference_rgb, width, height, zoom, pan_x, pan_y, rotation)
        vignette = self._vignette(width, height)
        grain = rng.normal(0.0, 2.2, foreground.shape).astype(np.float32)
        composed = cv2.addWeighted(background, 0.3, foreground, 0.7, 0.0).astype(np.float32)
        composed = np.clip(composed * vignette[..., None] + grain, 0, 255).astype(np.uint8)
        return composed

    def _frame_count(self, request: GenerateRequest) -> int:
        duration = 2.0 if request.extra_params.get("preview_mode") else (request.duration or 4.0)
        duration = max(1.0, float(duration))
        return max(1, int(round(duration * float(request.fps))))

    def _target_resolution(
        self,
        reference_image: np.ndarray,
        requested: Optional[tuple[int, int]],
        quality: QualityPreset,
    ) -> tuple[int, int]:
        if requested:
            return (
                max(2, int(requested[0] // 2 * 2)),
                max(2, int(requested[1] // 2 * 2)),
            )

        ref_h, ref_w = reference_image.shape[:2]
        target_h = min(720, max(256, ref_h))
        target_w = int(target_h * (ref_w / max(1, ref_h)))
        target_w = min(1280, max(256, target_w))
        scale = {
            QualityPreset.DRAFT: 0.75,
            QualityPreset.STANDARD: 1.0,
            QualityPreset.HIGH: 1.1,
            QualityPreset.ULTRA: 1.2,
        }.get(quality, 1.0)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)
        return max(2, target_w // 2 * 2), max(2, target_h // 2 * 2)

    def _prompt_seed(self, prompt: str) -> int:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _motion_profile(self, prompt: str, quality: QualityPreset) -> Dict[str, float]:
        lowered = prompt.lower()
        profile = {
            "zoom": 0.06,
            "zoom_cycles": 1.0,
            "pan_x": 14.0,
            "pan_y": 10.0,
            "rotation": 1.2,
        }
        if any(word in lowered for word in ["run", "chase", "fast", "race"]):
            profile.update({"pan_x": 24.0, "pan_y": 14.0, "rotation": 2.2})
        elif any(word in lowered for word in ["dance", "spin", "twirl"]):
            profile.update({"zoom_cycles": 2.0, "rotation": 4.0})
        elif any(word in lowered for word in ["float", "drift", "dream", "cinematic"]):
            profile.update({"zoom": 0.1, "pan_x": 8.0, "pan_y": 12.0, "rotation": 0.8})

        if quality == QualityPreset.DRAFT:
            profile["zoom"] *= 0.7
        elif quality == QualityPreset.ULTRA:
            profile["zoom"] *= 1.15
        return profile

    def _apply_prompt_tint(self, image: np.ndarray, prompt: str) -> np.ndarray:
        lowered = prompt.lower()
        overlay = np.zeros_like(image, dtype=np.uint8)
        if any(word in lowered for word in ["night", "moon", "dark"]):
            overlay[:] = (16, 24, 62)
        elif any(word in lowered for word in ["sunset", "warm", "golden"]):
            overlay[:] = (84, 48, 10)
        elif any(word in lowered for word in ["forest", "nature", "green"]):
            overlay[:] = (22, 54, 28)
        elif any(word in lowered for word in ["cyber", "neon", "city"]):
            overlay[:] = (54, 16, 60)
        else:
            return image
        return cv2.addWeighted(image, 0.78, overlay, 0.22, 0.0)

    def _transform_reference(
        self,
        reference_rgb: np.ndarray,
        width: int,
        height: int,
        zoom: float,
        pan_x: float,
        pan_y: float,
        rotation: float,
    ) -> np.ndarray:
        canvas = self._cover_resize(reference_rgb, width, height)
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0 + zoom)
        matrix[0, 2] += pan_x
        matrix[1, 2] += pan_y
        return cv2.warpAffine(
            canvas,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _cover_resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        scale = max(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(2, int(image.shape[1] * scale)), max(2, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        x1 = max(0, (resized.shape[1] - width) // 2)
        y1 = max(0, (resized.shape[0] - height) // 2)
        return resized[y1:y1 + height, x1:x1 + width]

    def _vignette(self, width: int, height: int) -> np.ndarray:
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        distance = np.sqrt(xv ** 2 + yv ** 2)
        return np.clip(1.08 - distance * 0.35, 0.72, 1.05)

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _to_bgr_pil(self, image: np.ndarray):
        from PIL import Image

        rgb = self._to_rgb(image)
        return Image.fromarray(rgb)

    def _write_video(self, frames: list[np.ndarray], output_path: Path, fps: float) -> None:
        if not frames:
            raise VideoProcessingError("No frames generated for Wan output")
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise VideoProcessingError(f"Could not open video writer for {output_path}")
        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()


_wan_service = None


def get_wan_video_service() -> WanVideoService:
    global _wan_service
    if _wan_service is None:
        _wan_service = WanVideoService()
    return _wan_service
