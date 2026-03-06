"""
Realtime frame processing pipeline shared by websocket and worker processes.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
from loguru import logger

from backend.core.models import GenerationMode
from backend.services.realtime.jpeg_codec import get_jpeg_codec


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
) -> bytes:
    """Process a single frame through the realtime pipeline."""
    codec = get_jpeg_codec()
    config = session["config"]

    frame_bgr = codec.decode(frame_data)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mode = processor["mode"]
    result = frame_rgb

    try:
        if mode == GenerationMode.LIVEPORTRAIT:
            if processor.get("model"):
                result = processor["model"].process_frame(
                    processor["reference_image"],
                    frame_rgb,
                    config,
                )
        elif mode == GenerationMode.DEEP_LIVE_CAM:
            if processor.get("model"):
                result = await processor["model"].swap_face(
                    processor["reference_image"],
                    frame_rgb,
                    config,
                )
    except Exception as exc:
        logger.error(f"Frame processing error: {exc}")
        result = frame_rgb

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    output_resolution = tuple(config.get("output_resolution") or ())
    if len(output_resolution) == 2:
        current_resolution = (result_bgr.shape[1], result_bgr.shape[0])
        if current_resolution != output_resolution:
            result_bgr = cv2.resize(result_bgr, output_resolution, interpolation=cv2.INTER_LINEAR)

    return codec.encode(
        result_bgr,
        quality=config.get("jpeg_quality", 90),
        subsampling=config.get("jpeg_subsampling", "420"),
    )


async def cleanup_processor(processor: Dict[str, Any]) -> None:
    """Clean up processor resources."""
    if processor.get("model"):
        if hasattr(processor["model"], "clear_source_cache"):
            processor["model"].clear_source_cache()
        if hasattr(processor["model"], "close"):
            processor["model"].close()

    if processor.get("face_detector"):
        processor["face_detector"].close()
