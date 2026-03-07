"""
JPEG codec abstraction for realtime transport.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from loguru import logger

try:
    from turbojpeg import (
        TurboJPEG,
        TJPF_BGR,
        TJSAMP_420,
        TJSAMP_422,
        TJSAMP_444,
        TJSAMP_GRAY,
    )
except ImportError:  # pragma: no cover - optional acceleration
    TurboJPEG = None
    TJPF_BGR = None
    TJSAMP_420 = None
    TJSAMP_422 = None
    TJSAMP_444 = None
    TJSAMP_GRAY = None


_SUBSAMPLING_MAP = {
    "444": TJSAMP_444,
    "422": TJSAMP_422,
    "420": TJSAMP_420,
    "gray": TJSAMP_GRAY,
}


class JpegCodec:
    """Encode and decode JPEG frames with TurboJPEG when available."""

    def __init__(self):
        self._turbo: Optional[TurboJPEG] = None
        if TurboJPEG is not None:
            try:
                self._turbo = TurboJPEG()
                logger.info("Realtime JPEG codec using TurboJPEG")
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning(f"TurboJPEG unavailable, falling back to OpenCV: {exc}")

        if self._turbo is None:
            logger.info("Realtime JPEG codec using OpenCV fallback")

    def decode(self, payload: bytes) -> np.ndarray:
        """Decode JPEG bytes to a BGR frame."""
        if self._turbo is not None:
            return self._turbo.decode(payload, pixel_format=TJPF_BGR)

        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode JPEG frame")
        return frame

    def encode(
        self,
        image_bgr: np.ndarray,
        *,
        quality: int = 90,
        subsampling: str = "420",
    ) -> bytes:
        """Encode a BGR frame to JPEG bytes."""
        quality = max(50, min(int(quality), 100))
        normalized_subsampling = subsampling.lower()

        if self._turbo is not None:
            jpeg_subsample = _SUBSAMPLING_MAP.get(normalized_subsampling, TJSAMP_420)
            return self._turbo.encode(
                image_bgr,
                quality=quality,
                pixel_format=TJPF_BGR,
                jpeg_subsample=jpeg_subsample,
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            raise ValueError("Failed to encode JPEG frame")
        return encoded.tobytes()


_jpeg_codec: Optional[JpegCodec] = None


def get_jpeg_codec() -> JpegCodec:
    """Get the process-local realtime JPEG codec singleton."""
    global _jpeg_codec
    if _jpeg_codec is None:
        _jpeg_codec = JpegCodec()
    return _jpeg_codec
