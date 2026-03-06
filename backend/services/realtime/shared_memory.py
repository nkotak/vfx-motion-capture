"""
Shared-memory helpers for large realtime JPEG payloads.
"""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any, Dict, Tuple


def create_shared_memory_payload(payload: bytes) -> tuple[dict[str, Any], shared_memory.SharedMemory]:
    """Create a shared-memory block containing the given payload."""
    size = max(1, len(payload))
    shm = shared_memory.SharedMemory(create=True, size=size)
    shm.buf[:len(payload)] = payload
    return {
        "name": shm.name,
        "size": len(payload),
    }, shm


def read_shared_memory_payload(
    payload_ref: Dict[str, Any],
    *,
    unlink: bool = False,
) -> bytes:
    """Read a payload from a shared-memory block."""
    shm = shared_memory.SharedMemory(name=payload_ref["name"])
    try:
        size = int(payload_ref["size"])
        return bytes(shm.buf[:size])
    finally:
        shm.close()
        if unlink:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def close_shared_memory(shm: shared_memory.SharedMemory, *, unlink: bool = False) -> None:
    """Close an opened shared-memory handle."""
    try:
        shm.close()
    finally:
        if unlink:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def cleanup_shared_memory_payload(payload_ref: Dict[str, Any]) -> None:
    """Close and unlink a shared-memory block by reference."""
    shm = shared_memory.SharedMemory(name=payload_ref["name"])
    close_shared_memory(shm, unlink=True)
