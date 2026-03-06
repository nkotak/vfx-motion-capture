"""
Adaptive realtime quality policy helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.core.models import GenerationMode
from backend.services.realtime.metrics import record_adaptive_event, sync_config_metrics


def initialize_adaptive_state(session: Dict[str, Any]) -> None:
    """Initialize adaptive state for a realtime session."""
    session.setdefault("requested_config", dict(session["config"]))
    session.setdefault("adaptive_state", {
        "last_adjustment_frame": 0,
        "last_direction": None,
    })


def maybe_apply_adaptive_quality(session: Dict[str, Any]) -> Optional[str]:
    """Adapt realtime session config when latency persistently drifts away from budget."""
    config = session["config"]
    metrics = session["metrics"]
    adaptive_state = session.setdefault("adaptive_state", {})
    requested_config = session.setdefault("requested_config", dict(config))

    sync_config_metrics(metrics, config)
    if not config.get("adaptive_quality", True):
        return None

    processed_frames = int(metrics.get("processed_frames", 0))
    cooldown_frames = int(config.get("adaptive_cooldown_frames") or 24)
    last_adjustment_frame = int(adaptive_state.get("last_adjustment_frame", 0))
    if processed_frames - last_adjustment_frame < cooldown_frames:
        return None

    target_fps = max(1, int(config.get("target_fps", 24)))
    derived_budget = max(1.0, 1000.0 / target_fps)
    latency_budget = float(config.get("adaptive_latency_budget_ms") or derived_budget)
    last_latency = float(metrics.get("last_total_latency_ms", 0.0))

    degrade_threshold = latency_budget * 1.2
    recover_threshold = latency_budget * 0.75

    event = None
    if last_latency > degrade_threshold:
        event = _degrade_config(config, requested_config)
        if event:
            adaptive_state["last_adjustment_frame"] = processed_frames
            adaptive_state["last_direction"] = "degrade"
    elif last_latency < recover_threshold:
        event = _recover_config(config, requested_config)
        if event:
            adaptive_state["last_adjustment_frame"] = processed_frames
            adaptive_state["last_direction"] = "recover"

    if event:
        record_adaptive_event(metrics, event, config)
    return event


def _degrade_config(config: Dict[str, Any], requested_config: Dict[str, Any]) -> Optional[str]:
    jpeg_step = int(config.get("adaptive_jpeg_step") or 5)
    min_jpeg = int(config.get("adaptive_min_jpeg_quality") or 75)
    current_quality = int(config.get("jpeg_quality", requested_config.get("jpeg_quality", 90)))
    if current_quality > min_jpeg:
        config["jpeg_quality"] = max(min_jpeg, current_quality - jpeg_step)
        return f"Reduced JPEG quality to {config['jpeg_quality']}"

    mode = GenerationMode(config["mode"])
    adaptive_tile_size = int(config.get("adaptive_tile_size") or 0)
    adaptive_min_tile_size = int(config.get("adaptive_min_tile_size") or 512)
    current_tile_size = int(config.get("tile_size") or 0)

    if mode == GenerationMode.LIVEPORTRAIT and adaptive_tile_size > 0:
        if current_tile_size <= 0:
            config["tile_size"] = adaptive_tile_size
            return f"Enabled tiled full-frame path at {adaptive_tile_size}px"
        if current_tile_size > adaptive_min_tile_size:
            next_tile_size = max(adaptive_min_tile_size, current_tile_size // 2)
            if next_tile_size != current_tile_size:
                config["tile_size"] = next_tile_size
                return f"Reduced tile size to {next_tile_size}px"

    if config.get("full_frame_inference", True):
        config["full_frame_inference"] = False
        return "Disabled full-frame inference fallback"

    return None


def _recover_config(config: Dict[str, Any], requested_config: Dict[str, Any]) -> Optional[str]:
    requested_quality = int(requested_config.get("jpeg_quality", config.get("jpeg_quality", 90)))
    jpeg_step = int(config.get("adaptive_jpeg_step") or 5)
    current_quality = int(config.get("jpeg_quality", requested_quality))
    if current_quality < requested_quality:
        config["jpeg_quality"] = min(requested_quality, current_quality + jpeg_step)
        return f"Raised JPEG quality to {config['jpeg_quality']}"

    mode = GenerationMode(config["mode"])
    requested_tile_size = int(requested_config.get("tile_size") or 0)
    current_tile_size = int(config.get("tile_size") or 0)
    if mode == GenerationMode.LIVEPORTRAIT and current_tile_size > requested_tile_size:
        if requested_tile_size > 0:
            next_tile_size = min(current_tile_size * 2, requested_tile_size)
            config["tile_size"] = next_tile_size
            return f"Increased tile size to {next_tile_size}px"
        config["tile_size"] = requested_config.get("tile_size")
        return "Disabled tiled full-frame path"

    requested_full_frame = bool(requested_config.get("full_frame_inference", True))
    if requested_full_frame and not config.get("full_frame_inference", False):
        config["full_frame_inference"] = True
        return "Re-enabled full-frame inference"

    return None
