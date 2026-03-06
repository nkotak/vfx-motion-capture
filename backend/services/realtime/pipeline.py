"""
Realtime frame processing pipeline shared by websocket and worker processes.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np
from loguru import logger

from backend.core.models import GenerationMode
from backend.services.realtime.jpeg_codec import get_jpeg_codec
from backend.services.realtime.tiled_inference import _axis_weights, _positions


def _determine_liveportrait_tile_size(config: Dict[str, Any], frame_rgb: np.ndarray) -> int:
    """Choose the liveportrait tile size for the current frame."""
    explicit_tile_size = int(config.get("tile_size") or 0)
    if explicit_tile_size > 0:
        return explicit_tile_size

    if not config.get("full_frame_inference", True):
        return 0

    height, width = frame_rgb.shape[:2]
    if max(height, width) >= 2560:
        return int(config.get("adaptive_tile_size") or 1024)
    return 0


async def _resolve_liveportrait_result(
    processor: Dict[str, Any],
    frame_rgb: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Run the liveportrait frame processor and await it if needed."""
    result = processor["model"].process_frame(
        processor["reference_image"],
        frame_rgb,
        config,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def _process_liveportrait_tiled(
    processor: Dict[str, Any],
    frame_rgb: np.ndarray,
    config: Dict[str, Any],
    *,
    tile_size: int,
    tile_overlap: int,
) -> np.ndarray:
    """Apply liveportrait to a frame tile-by-tile and blend the result."""
    height, width = frame_rgb.shape[:2]
    if tile_size <= 0 or max(height, width) <= tile_size:
        return await _resolve_liveportrait_result(processor, frame_rgb, config)

    ys = _positions(height, tile_size, tile_overlap)
    xs = _positions(width, tile_size, tile_overlap)
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight_map = np.zeros((height, width, 1), dtype=np.float32)

    for y in ys:
        for x in xs:
            y2 = min(height, y + tile_size)
            x2 = min(width, x + tile_size)
            tile = frame_rgb[y:y2, x:x2]
            processed_tile = await _resolve_liveportrait_result(processor, tile, config)
            if processed_tile.shape != tile.shape:
                raise ValueError("LivePortrait tiled processing must preserve tile dimensions")

            y_weights = _axis_weights(y2 - y, tile_overlap, y == 0, y2 == height)
            x_weights = _axis_weights(x2 - x, tile_overlap, x == 0, x2 == width)
            blend = np.outer(y_weights, x_weights)[..., None]

            accum[y:y2, x:x2] += processed_tile.astype(np.float32) * blend
            weight_map[y:y2, x:x2] += blend

    return (accum / np.clip(weight_map, 1e-6, None)).astype(frame_rgb.dtype)


async def initialize_realtime_processor(session: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the real-time processor based on session config."""
    from backend.services.inference.face_swapper import get_face_swapper
    from backend.services.inference.live_portrait import get_live_portrait_service

    config = session["config"]
    mode = GenerationMode(config["mode"])
    reference_path = session["reference_path"]

    reference_bgr = cv2.imread(reference_path)
    if reference_bgr is None:
        raise ValueError(f"Failed to load reference image: {reference_path}")

    reference_image = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    processor = {
        "mode": mode,
        "reference_image": reference_image,
        "config": config,
        "model": None,
    }

    if mode == GenerationMode.LIVEPORTRAIT:
        logger.info("Initializing LivePortrait processor")
        processor["model"] = get_live_portrait_service()
    elif mode == GenerationMode.DEEP_LIVE_CAM:
        logger.info("Initializing face swap processor")
        face_swapper = get_face_swapper()
        await face_swapper.set_source_face(reference_image)
        processor["model"] = face_swapper

    return processor


async def process_frame(
    processor: Dict[str, Any],
    frame_data: bytes,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    """Process a single frame through the realtime pipeline."""
    codec = get_jpeg_codec()
    config = session["config"]
    stage_metrics = {
        "decode_ms": 0.0,
        "inference_ms": 0.0,
        "resize_ms": 0.0,
        "encode_ms": 0.0,
        "tile_count": 0,
    }

    decode_start = time.perf_counter()
    frame_bgr = codec.decode(frame_data)
    stage_metrics["decode_ms"] = (time.perf_counter() - decode_start) * 1000
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mode = processor["mode"]
    result = frame_rgb

    try:
        inference_start = time.perf_counter()
        if mode == GenerationMode.LIVEPORTRAIT:
            if processor.get("model"):
                tile_size = _determine_liveportrait_tile_size(config, frame_rgb)
                if tile_size > 0:
                    result = await _process_liveportrait_tiled(
                        processor,
                        frame_rgb,
                        config,
                        tile_size=tile_size,
                        tile_overlap=int(config.get("tile_overlap", 64)),
                    )
                    stage_metrics["tile_count"] = len(_positions(frame_rgb.shape[0], tile_size, int(config.get("tile_overlap", 64)))) * len(_positions(frame_rgb.shape[1], tile_size, int(config.get("tile_overlap", 64))))
                    stage_metrics["processing_mode"] = "liveportrait_tiled"
                else:
                    result = await _resolve_liveportrait_result(
                        processor,
                        frame_rgb,
                        config,
                    )
                    stage_metrics["processing_mode"] = "liveportrait_full_frame"
        elif mode == GenerationMode.DEEP_LIVE_CAM:
            if processor.get("model"):
                result = await processor["model"].swap_face(
                    processor["reference_image"],
                    frame_rgb,
                    config,
                )
                stage_metrics["processing_mode"] = (
                    "deep_live_cam_full_frame"
                    if config.get("full_frame_inference", True)
                    else "deep_live_cam_optimized_detection"
                )
        stage_metrics["inference_ms"] = (time.perf_counter() - inference_start) * 1000
    except Exception as exc:
        logger.error(f"Frame processing error: {exc}")
        stage_metrics["inference_ms"] = (time.perf_counter() - inference_start) * 1000
        result = frame_rgb
        stage_metrics["processing_mode"] = "passthrough_error"

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    output_resolution = tuple(config.get("output_resolution") or ())
    if len(output_resolution) == 2:
        current_resolution = (result_bgr.shape[1], result_bgr.shape[0])
        if current_resolution != output_resolution:
            resize_start = time.perf_counter()
            result_bgr = cv2.resize(result_bgr, output_resolution, interpolation=cv2.INTER_LINEAR)
            stage_metrics["resize_ms"] = (time.perf_counter() - resize_start) * 1000

    encode_start = time.perf_counter()
    encoded = codec.encode(
        result_bgr,
        quality=config.get("jpeg_quality", 90),
        subsampling=config.get("jpeg_subsampling", "420"),
    )
    stage_metrics["encode_ms"] = (time.perf_counter() - encode_start) * 1000

    return {
        "frame_data": encoded,
        "metrics": stage_metrics,
    }


async def cleanup_processor(processor: Dict[str, Any]) -> None:
    """Clean up processor resources."""
    if processor.get("model"):
        if hasattr(processor["model"], "clear_source_cache"):
            processor["model"].clear_source_cache()
        if hasattr(processor["model"], "close"):
            processor["model"].close()

    if processor.get("face_detector"):
        processor["face_detector"].close()
