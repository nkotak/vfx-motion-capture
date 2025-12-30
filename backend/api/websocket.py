"""
WebSocket handlers for real-time communication.
"""

import asyncio
import json
import base64
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from backend.core.models import JobStatus, JobProgress
from backend.services.job_manager import get_job_manager


# Router for WebSocket endpoints
websocket_router = APIRouter()

# Store for active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Store for real-time sessions
realtime_sessions: Dict[str, Dict[str, Any]] = {}


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)
        logger.debug(f"WebSocket connected: {channel}")

    def disconnect(self, websocket: WebSocket, channel: str):
        """Unregister a WebSocket connection."""
        if channel in self.active_connections:
            try:
                self.active_connections[channel].remove(websocket)
                if not self.active_connections[channel]:
                    del self.active_connections[channel]
            except ValueError:
                pass
        logger.debug(f"WebSocket disconnected: {channel}")

    async def send_to_channel(self, channel: str, message: dict):
        """Send a message to all connections in a channel."""
        if channel not in self.active_connections:
            return

        dead_connections = []
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(conn, channel)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connections."""
        for channel in list(self.active_connections.keys()):
            await self.send_to_channel(channel, message)


manager = ConnectionManager()


@websocket_router.websocket("/ws/jobs/{job_id}")
async def job_progress_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates.

    Connect to receive progress updates for a specific job.
    Messages are JSON with format:
    {
        "type": "progress" | "complete" | "error",
        "data": {...}
    }
    """
    job_manager = get_job_manager()

    # Verify job exists
    try:
        job = job_manager.get_job(job_id)
    except Exception:
        await websocket.close(code=4004, reason="Job not found")
        return

    channel = f"job:{job_id}"
    await manager.connect(websocket, channel)

    # Register callback for progress updates
    async def progress_callback(progress: JobProgress):
        await manager.send_to_channel(channel, {
            "type": "progress",
            "data": progress.model_dump(),
        })

    job_manager.register_callback(job_id, progress_callback)

    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "data": {
                "job_id": job_id,
                "status": job.status.value,
                "progress": job.progress,
            },
        })

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for any message (can be used for ping/pong)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await websocket.send_json({"type": "keepalive"})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket error for job {job_id}: {e}")
    finally:
        job_manager.unregister_callback(job_id, progress_callback)
        manager.disconnect(websocket, channel)


@websocket_router.websocket("/ws/realtime/{session_id}")
async def realtime_processing_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time camera processing.

    ## Protocol

    1. Client sends camera frames as base64-encoded images
    2. Server processes and returns transformed frames
    3. Binary frames can also be used for efficiency

    ## Message Format

    Client -> Server:
    {
        "type": "frame",
        "data": "<base64 image data>",
        "timestamp": 1234567890
    }

    Server -> Client:
    {
        "type": "result",
        "data": "<base64 transformed image>",
        "latency_ms": 25,
        "timestamp": 1234567890
    }
    """
    # Verify session exists
    if session_id not in realtime_sessions:
        await websocket.close(code=4004, reason="Session not found")
        return

    session = realtime_sessions[session_id]
    await websocket.accept()

    logger.info(f"Real-time session started: {session_id}")
    session["status"] = "active"
    session["created_at"] = datetime.utcnow()

    # Initialize processor based on mode
    processor = await initialize_realtime_processor(session)

    frame_count = 0
    total_latency = 0

    try:
        while True:
            # Receive frame
            message = await websocket.receive()

            if "text" in message:
                data = json.loads(message["text"])

                if data.get("type") == "frame":
                    start_time = asyncio.get_event_loop().time()

                    # Decode frame
                    frame_data = base64.b64decode(data["data"])

                    # Process frame
                    result_data = await process_frame(processor, frame_data, session)

                    # Calculate latency
                    latency = (asyncio.get_event_loop().time() - start_time) * 1000

                    frame_count += 1
                    total_latency += latency

                    # Send result
                    await websocket.send_json({
                        "type": "result",
                        "data": base64.b64encode(result_data).decode(),
                        "latency_ms": round(latency, 1),
                        "frame_number": frame_count,
                        "timestamp": data.get("timestamp"),
                    })

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                elif data.get("type") == "stop":
                    break

            elif "bytes" in message:
                # Binary frame for efficiency
                start_time = asyncio.get_event_loop().time()

                result_data = await process_frame(processor, message["bytes"], session)

                latency = (asyncio.get_event_loop().time() - start_time) * 1000
                frame_count += 1
                total_latency += latency

                # Send binary result
                await websocket.send_bytes(result_data)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Real-time processing error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass
    finally:
        # Cleanup
        session["status"] = "closed"
        if processor:
            await cleanup_processor(processor)

        avg_latency = total_latency / frame_count if frame_count > 0 else 0
        logger.info(
            f"Real-time session ended: {session_id} "
            f"(frames: {frame_count}, avg latency: {avg_latency:.1f}ms)"
        )


async def initialize_realtime_processor(session: Dict[str, Any]) -> Any:
    """Initialize the real-time processor based on session config."""
    from backend.core.models import GenerationMode
    import cv2
    import numpy as np
    from PIL import Image

    config = session["config"]
    mode = GenerationMode(config["mode"])
    reference_path = session["reference_path"]

    # Load reference image
    reference_image = cv2.imread(reference_path)
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    processor = {
        "mode": mode,
        "reference_image": reference_image,
        "config": config,
        "model": None,
    }

    if mode == GenerationMode.LIVEPORTRAIT:
        # Initialize LivePortrait
        try:
            # Note: This is a placeholder - actual implementation would load
            # the LivePortrait model
            logger.info("Initializing LivePortrait processor")
            # from liveportrait import LivePortraitPipeline
            # processor["model"] = LivePortraitPipeline()
            # processor["model"].prepare(reference_image)
        except Exception as e:
            logger.warning(f"Failed to initialize LivePortrait: {e}")

    elif mode == GenerationMode.DEEP_LIVE_CAM:
        # Initialize Deep Live Cam / face swapper
        try:
            logger.info("Initializing face swap processor")
            from backend.services.face_detector import get_face_detector

            face_detector = get_face_detector()
            await face_detector.initialize()

            # Extract source face embedding
            source_face = await face_detector.get_primary_face(reference_image)
            processor["source_face"] = source_face
            processor["face_detector"] = face_detector
        except Exception as e:
            logger.warning(f"Failed to initialize face detector: {e}")

    return processor


async def process_frame(
    processor: Dict[str, Any],
    frame_data: bytes,
    session: Dict[str, Any]
) -> bytes:
    """Process a single frame through the real-time pipeline."""
    import cv2
    import numpy as np
    from backend.core.models import GenerationMode

    # Decode frame
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Failed to decode frame")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mode = processor["mode"]
    result = frame_rgb  # Default to passthrough

    if mode == GenerationMode.LIVEPORTRAIT:
        # Apply LivePortrait transformation
        if processor.get("model"):
            # result = processor["model"].animate(frame_rgb)
            pass
        else:
            # Fallback: simple color adjustment to show processing
            result = cv2.addWeighted(
                frame_rgb, 0.7,
                processor["reference_image"][:frame_rgb.shape[0], :frame_rgb.shape[1]],
                0.3,
                0
            ) if processor["reference_image"].shape[:2] >= frame_rgb.shape[:2] else frame_rgb

    elif mode == GenerationMode.DEEP_LIVE_CAM:
        # Apply face swap
        face_detector = processor.get("face_detector")
        if face_detector:
            try:
                # Detect face in current frame
                faces = await face_detector.detect_faces(frame_rgb, max_faces=1)
                if faces:
                    # In a full implementation, we would:
                    # 1. Align the source face to the target position
                    # 2. Blend the faces together
                    # 3. Apply color correction
                    # For now, just overlay the reference
                    source_face = processor.get("source_face")
                    if source_face:
                        # Draw indicator that processing is happening
                        result = face_detector.draw_faces(frame_rgb, faces)
            except Exception as e:
                logger.warning(f"Face processing error: {e}")

    # Encode result
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return encoded.tobytes()


async def cleanup_processor(processor: Dict[str, Any]) -> None:
    """Clean up processor resources."""
    if processor.get("model"):
        if hasattr(processor["model"], "close"):
            processor["model"].close()

    if processor.get("face_detector"):
        processor["face_detector"].close()


@websocket_router.websocket("/ws/status")
async def status_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for general system status updates.

    Broadcasts:
    - New job notifications
    - System health updates
    - Queue status changes
    """
    await manager.connect(websocket, "status")

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to status stream",
        })

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send status update
                job_manager = get_job_manager()
                stats = job_manager.get_stats()

                await websocket.send_json({
                    "type": "status",
                    "data": stats,
                })

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, "status")
