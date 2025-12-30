"""
Hardware/device resolution helpers.

We allow users to set `settings.device` to "auto". In that case we detect the
best available accelerator at runtime and return a concrete device string.
"""

from __future__ import annotations

from typing import Literal, Optional

from loguru import logger

ResolvedDevice = Literal["cuda", "mps", "cpu"]


def resolve_device(device: Optional[str]) -> ResolvedDevice:
    """
    Resolve a user-configured device value into a concrete device.

    Accepted inputs: "auto", "cuda", "mps", "cpu" (case-insensitive).
    Unknown values are treated as "auto" with a warning.
    """
    requested = (device or "auto").strip().lower()
    if requested in ("cuda", "mps", "cpu"):
        return requested  # type: ignore[return-value]

    if requested != "auto":
        logger.warning(f"Unknown device '{device}', falling back to auto-detect")

    # Auto-detect. Import torch lazily to avoid making it a hard import-time
    # dependency for codepaths that don't use ML.
    try:
        import torch
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not import torch for device auto-detect ({e}); using CPU")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"

