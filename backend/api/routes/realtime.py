"""
Real-time video processing endpoints.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.core.config import settings
from backend.core.models import RealtimeConfig, GenerationMode
from backend.core.exceptions import FileNotFoundError
from backend.services.file_manager import get_file_manager
from backend.services.realtime import get_realtime_worker_pool


router = APIRouter()


@router.post("/realtime/session")
async def create_realtime_session(config: RealtimeConfig):
    """
    Create a new real-time processing session.

    This initializes a session for real-time camera-based motion capture.
    The returned session_id is used to connect via WebSocket.

    ## Modes

    - **liveportrait**: Animate portrait with facial expressions (fastest)
    - **deep_live_cam**: Full face swap in real-time

    ## Example

    ```json
    {
        "reference_image_id": "abc123",
        "mode": "liveportrait",
        "target_fps": 30,
        "face_only": false
    }
    ```
    """
    file_manager = get_file_manager()

    # Validate reference image
    try:
        ref_image = file_manager.get_file(config.reference_image_id)
        if ref_image.file_type != "image":
            raise HTTPException(
                status_code=400,
                detail="Reference must be an image"
            )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Reference image not found: {config.reference_image_id}"
        )

    # Validate mode is suitable for real-time
    if config.mode not in [GenerationMode.LIVEPORTRAIT, GenerationMode.DEEP_LIVE_CAM]:
        raise HTTPException(
            status_code=400,
            detail=f"Mode {config.mode} is not supported for real-time processing. "
                   f"Use 'liveportrait' or 'deep_live_cam'."
        )

    normalized_output_resolution = config.output_resolution or config.input_resolution
    config_payload = config.model_copy(
        update={
            "output_resolution": normalized_output_resolution,
            "binary_transport": True if settings.realtime_binary_transport else config.binary_transport,
            "allow_frame_drop": config.allow_frame_drop if config.allow_frame_drop is not None else settings.realtime_allow_frame_drop,
            "max_inflight_frames": config.max_inflight_frames or settings.realtime_max_inflight_frames,
            "full_frame_inference": config.full_frame_inference if config.full_frame_inference is not None else settings.realtime_full_frame_inference,
        }
    )

    # Create session
    import uuid
    session_id = str(uuid.uuid4())

    # Store session config (in production, use Redis or similar)
    from backend.api.websocket import realtime_sessions
    realtime_sessions[session_id] = {
        "config": config_payload.model_dump(),
        "reference_path": str(ref_image.path),
        "status": "ready",
        "created_at": None,
    }
    try:
        worker_id = await get_realtime_worker_pool().register_session(
            session_id,
            realtime_sessions[session_id],
        )
        realtime_sessions[session_id]["worker_id"] = worker_id
    except Exception as exc:
        realtime_sessions.pop(session_id, None)
        logger.error(f"Failed to initialize realtime session worker: {exc}")
        raise HTTPException(status_code=500, detail="Failed to initialize realtime worker") from exc

    logger.info(f"Created real-time session: {session_id} (mode: {config.mode})")

    return {
        "session_id": session_id,
        "websocket_url": f"/ws/realtime/{session_id}",
        "config": config_payload.model_dump(),
        "status": "ready",
        "worker_id": worker_id,
    }


@router.get("/realtime/session/{session_id}")
async def get_session_info(session_id: str):
    """
    Get information about a real-time session.
    """
    from backend.api.websocket import realtime_sessions

    if session_id not in realtime_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = realtime_sessions[session_id]

    return {
        "session_id": session_id,
        "config": session["config"],
        "status": session["status"],
        "websocket_url": f"/ws/realtime/{session_id}",
        "worker_id": session.get("worker_id"),
    }


@router.delete("/realtime/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a real-time session.
    """
    from backend.api.websocket import realtime_sessions

    if session_id not in realtime_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    await get_realtime_worker_pool().close_session(session_id)
    del realtime_sessions[session_id]
    logger.info(f"Deleted real-time session: {session_id}")

    return {"status": "deleted", "session_id": session_id}


@router.get("/realtime/modes")
async def list_realtime_modes():
    """
    List available real-time processing modes.
    """
    return {
        "modes": [
            {
                "value": GenerationMode.LIVEPORTRAIT.value,
                "name": "LivePortrait",
                "description": "Real-time portrait animation using facial expressions. "
                              "Fastest option, best for face-focused content.",
                "latency": "~15-30ms",
                "requirements": "8GB VRAM",
            },
            {
                "value": GenerationMode.DEEP_LIVE_CAM.value,
                "name": "Deep Live Cam",
                "description": "Real-time face swap. Replaces your face with the "
                              "reference character while maintaining your expressions.",
                "latency": "~30-50ms",
                "requirements": "8GB VRAM",
            },
        ]
    }


@router.get("/realtime/check-compatibility")
async def check_compatibility():
    """
    Check if the system supports real-time processing.

    Returns information about GPU availability and estimated performance.
    """
    import torch

    gpu_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    gpu_name = None
    gpu_memory = None
    runtime = "cpu"

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        runtime = "cuda"
    elif mps_available:
        gpu_name = "Apple Silicon (MPS)"
        runtime = "mps"

    # Estimate capability
    capability = "none"
    estimated_fps = 0

    if gpu_available and gpu_memory:
        if gpu_memory >= 24:
            capability = "excellent"
            estimated_fps = 60
        elif gpu_memory >= 12:
            capability = "good"
            estimated_fps = 30
        elif gpu_memory >= 8:
            capability = "moderate"
            estimated_fps = 20
        elif gpu_memory >= 4:
            capability = "limited"
            estimated_fps = 10
    elif mps_available:
        capability = "good"
        estimated_fps = 24

    return {
        "gpu_available": gpu_available or mps_available,
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_memory, 1) if gpu_memory else None,
        "capability": capability,
        "estimated_fps": estimated_fps,
        "runtime": runtime,
        "recommended_mode": (
            GenerationMode.LIVEPORTRAIT.value if capability in ["excellent", "good"]
            else GenerationMode.DEEP_LIVE_CAM.value if capability == "moderate"
            else None
        ),
    }
