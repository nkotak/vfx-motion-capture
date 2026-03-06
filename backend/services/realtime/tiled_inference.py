"""
Helpers for tiled full-frame processing.
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np


def _positions(size: int, tile_size: int, overlap: int) -> list[int]:
    step = max(1, tile_size - overlap)
    positions = list(range(0, max(size - tile_size, 0) + 1, step))
    if not positions:
        return [0]
    last = max(0, size - tile_size)
    if positions[-1] != last:
        positions.append(last)
    return positions


def _axis_weights(length: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    fade = min(overlap, length // 2)
    if fade <= 0:
        return weights

    if not at_start:
        weights[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    if not at_end:
        weights[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
    return weights


def process_frame_tiled(
    frame_rgb: np.ndarray,
    *,
    tile_size: int,
    tile_overlap: int,
    processor: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, int]:
    """Process a frame tile-by-tile and blend the output back together."""
    height, width = frame_rgb.shape[:2]
    if tile_size <= 0 or max(height, width) <= tile_size:
        return processor(frame_rgb), 1

    ys = _positions(height, tile_size, tile_overlap)
    xs = _positions(width, tile_size, tile_overlap)

    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight_map = np.zeros((height, width, 1), dtype=np.float32)
    tile_count = 0

    for y in ys:
        for x in xs:
            y2 = min(height, y + tile_size)
            x2 = min(width, x + tile_size)
            tile = frame_rgb[y:y2, x:x2]
            processed_tile = processor(tile)
            if processed_tile.shape != tile.shape:
                raise ValueError("Tiled processor must preserve tile dimensions")

            y_weights = _axis_weights(y2 - y, tile_overlap, y == 0, y2 == height)
            x_weights = _axis_weights(x2 - x, tile_overlap, x == 0, x2 == width)
            blend = np.outer(y_weights, x_weights)[..., None]

            accum[y:y2, x:x2] += processed_tile.astype(np.float32) * blend
            weight_map[y:y2, x:x2] += blend
            tile_count += 1

    blended = accum / np.clip(weight_map, 1e-6, None)
    return blended.astype(frame_rgb.dtype), tile_count
