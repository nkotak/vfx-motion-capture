"""
Realtime session metrics helpers.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional


def create_realtime_metrics() -> Dict[str, Any]:
    """Create an empty realtime metrics payload."""
    return {
        "received_frames": 0,
        "processed_frames": 0,
        "dropped_frames": 0,
        "bytes_in": 0,
        "bytes_out": 0,
        "avg_worker_latency_ms": 0.0,
        "last_worker_latency_ms": 0.0,
        "avg_total_latency_ms": 0.0,
        "last_total_latency_ms": 0.0,
        "avg_decode_ms": 0.0,
        "last_decode_ms": 0.0,
        "avg_inference_ms": 0.0,
        "last_inference_ms": 0.0,
        "avg_encode_ms": 0.0,
        "last_encode_ms": 0.0,
        "avg_resize_ms": 0.0,
        "last_resize_ms": 0.0,
        "avg_tile_count": 0.0,
        "last_tile_count": 0,
        "shared_memory_in_count": 0,
        "shared_memory_in_bytes": 0,
        "shared_memory_out_count": 0,
        "shared_memory_out_bytes": 0,
        "inline_transport_in_count": 0,
        "inline_transport_in_bytes": 0,
        "inline_transport_out_count": 0,
        "inline_transport_out_bytes": 0,
        "adaptive_adjustment_count": 0,
        "adaptive_events": [],
        "current_jpeg_quality": None,
        "current_tile_size": None,
        "current_full_frame_inference": None,
        "current_target_fps": None,
        "current_processing_mode": None,
        "worker_id": None,
        "last_updated_at": None,
    }


def record_received(metrics: Dict[str, Any], payload_size: int) -> None:
    """Record an inbound frame."""
    metrics["received_frames"] += 1
    metrics["bytes_in"] += max(0, int(payload_size))
    metrics["last_updated_at"] = datetime.utcnow().isoformat()


def record_dropped(metrics: Dict[str, Any]) -> None:
    """Record a dropped stale frame."""
    metrics["dropped_frames"] += 1
    metrics["last_updated_at"] = datetime.utcnow().isoformat()


def _update_average(metrics: Dict[str, Any], avg_key: str, last_key: str, value: float, count: int) -> None:
    metrics[last_key] = float(value)
    if count <= 1:
        metrics[avg_key] = float(value)
    else:
        previous = float(metrics.get(avg_key, 0.0))
        metrics[avg_key] = previous + ((float(value) - previous) / count)


def record_processed(
    metrics: Dict[str, Any],
    *,
    output_bytes: int,
    worker_latency_ms: float,
    total_latency_ms: float,
    stage_metrics: Optional[Dict[str, Any]] = None,
    transport_metrics: Optional[Dict[str, Any]] = None,
    worker_id: Optional[int] = None,
) -> None:
    """Record a processed frame and update moving averages."""
    metrics["processed_frames"] += 1
    processed_frames = metrics["processed_frames"]
    metrics["bytes_out"] += max(0, int(output_bytes))
    if worker_id is not None:
        metrics["worker_id"] = worker_id

    _update_average(
        metrics,
        "avg_worker_latency_ms",
        "last_worker_latency_ms",
        worker_latency_ms,
        processed_frames,
    )
    _update_average(
        metrics,
        "avg_total_latency_ms",
        "last_total_latency_ms",
        total_latency_ms,
        processed_frames,
    )

    stage_metrics = stage_metrics or {}
    _update_average(
        metrics,
        "avg_decode_ms",
        "last_decode_ms",
        float(stage_metrics.get("decode_ms", 0.0)),
        processed_frames,
    )
    _update_average(
        metrics,
        "avg_inference_ms",
        "last_inference_ms",
        float(stage_metrics.get("inference_ms", 0.0)),
        processed_frames,
    )
    _update_average(
        metrics,
        "avg_encode_ms",
        "last_encode_ms",
        float(stage_metrics.get("encode_ms", 0.0)),
        processed_frames,
    )
    _update_average(
        metrics,
        "avg_resize_ms",
        "last_resize_ms",
        float(stage_metrics.get("resize_ms", 0.0)),
        processed_frames,
    )
    _update_average(
        metrics,
        "avg_tile_count",
        "last_tile_count",
        float(stage_metrics.get("tile_count", 0)),
        processed_frames,
    )
    if stage_metrics.get("processing_mode"):
        metrics["current_processing_mode"] = stage_metrics.get("processing_mode")

    transport_metrics = transport_metrics or {}
    input_bytes = int(transport_metrics.get("input_bytes", 0))
    output_bytes_metric = int(transport_metrics.get("output_bytes", output_bytes))
    if transport_metrics.get("input_shared_memory"):
        metrics["shared_memory_in_count"] += 1
        metrics["shared_memory_in_bytes"] += max(0, input_bytes)
    else:
        metrics["inline_transport_in_count"] += 1
        metrics["inline_transport_in_bytes"] += max(0, input_bytes)

    if transport_metrics.get("output_shared_memory"):
        metrics["shared_memory_out_count"] += 1
        metrics["shared_memory_out_bytes"] += max(0, output_bytes_metric)
    else:
        metrics["inline_transport_out_count"] += 1
        metrics["inline_transport_out_bytes"] += max(0, output_bytes_metric)
    metrics["last_updated_at"] = datetime.utcnow().isoformat()


def record_adaptive_event(metrics: Dict[str, Any], message: str, config: Dict[str, Any]) -> None:
    """Record an adaptive quality adjustment event."""
    metrics["adaptive_adjustment_count"] += 1
    events = metrics.setdefault("adaptive_events", [])
    events.append({
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
        "jpeg_quality": config.get("jpeg_quality"),
        "tile_size": config.get("tile_size"),
        "full_frame_inference": config.get("full_frame_inference"),
        "target_fps": config.get("target_fps"),
    })
    if len(events) > 10:
        del events[:-10]
    sync_config_metrics(metrics, config)
    metrics["last_updated_at"] = datetime.utcnow().isoformat()


def sync_config_metrics(metrics: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Mirror the effective realtime config into the metrics snapshot."""
    metrics["current_jpeg_quality"] = config.get("jpeg_quality")
    metrics["current_tile_size"] = config.get("tile_size")
    metrics["current_full_frame_inference"] = config.get("full_frame_inference")
    metrics["current_target_fps"] = config.get("target_fps")


def snapshot_realtime_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy safe for API responses."""
    return deepcopy(metrics)
