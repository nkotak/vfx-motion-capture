"""
WebSocket handlers for real-time communication.
"""

import asyncio
import json
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from backend.core.models import JobStatus, JobProgress
from backend.services.job_manager import get_job_manager
from backend.services.realtime.metrics import record_dropped, record_processed, record_received
from backend.services.realtime import get_realtime_worker_pool


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
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.warning(f"WebSocket error for job {job_id}: {e}")
    finally:
        job_manager.unregister_callback(job_id, progress_callback)
        manager.disconnect(websocket, channel)


@websocket_router.websocket("/ws/realtime/{session_id}")
async def realtime_processing_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time camera processing.

    Optimized path:
    - client sends binary JPEG frames
    - worker pool decodes / processes / re-encodes them
    - server streams binary JPEG frames back

    JSON text messages are reserved for control and error handling only.
    """
    if session_id not in realtime_sessions:
        await websocket.close(code=4004, reason="Session not found")
        return

    session = realtime_sessions[session_id]
    worker_pool = get_realtime_worker_pool()
    await worker_pool.register_session(session_id, session)
    await websocket.accept()

    logger.info(f"Real-time session started: {session_id}")
    session["status"] = "active"
    session["created_at"] = datetime.utcnow()

    metrics = session.setdefault("metrics", {})
    allow_frame_drop = bool(session["config"].get("allow_frame_drop", True))
    max_inflight_frames = max(1, int(session["config"].get("max_inflight_frames", 1)))
    frame_buffer: list[tuple[bytes, float]] = []
    frame_lock = asyncio.Lock()
    frame_ready = asyncio.Event()
    stop_event = asyncio.Event()
    send_lock = asyncio.Lock()
    loop = asyncio.get_running_loop()
    receiver_task: asyncio.Task | None = None
    processor_task: asyncio.Task | None = None

    async def send_json_safe(payload: Dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_bytes_safe(payload: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(payload)

    async def receiver_loop() -> None:
        while not stop_event.is_set():
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                stop_event.set()
                frame_ready.set()
                break

            if "text" in message and message["text"] is not None:
                data = json.loads(message["text"])

                if data.get("type") == "ping":
                    await send_json_safe({"type": "pong"})
                elif data.get("type") == "stop":
                    stop_event.set()
                    frame_ready.set()
                    break
                elif data.get("type") == "frame":
                    await send_json_safe({
                        "type": "error",
                        "message": "Realtime sessions now require binary JPEG websocket frames",
                    })
                continue

            payload = message.get("bytes")
            if payload is None:
                continue

            while not stop_event.is_set():
                async with frame_lock:
                    if len(frame_buffer) < max_inflight_frames:
                        record_received(metrics, len(payload))
                        frame_buffer.append((payload, loop.time()))
                        frame_ready.set()
                        break
                    if allow_frame_drop:
                        record_dropped(metrics)
                        frame_buffer.pop(0)
                        record_received(metrics, len(payload))
                        frame_buffer.append((payload, loop.time()))
                        frame_ready.set()
                        break

                await asyncio.sleep(0)

    async def processor_loop() -> None:
        while True:
            await frame_ready.wait()

            async with frame_lock:
                if not frame_buffer:
                    frame_ready.clear()
                    payload = None
                    received_at = None
                else:
                    if allow_frame_drop and len(frame_buffer) > 1:
                        dropped_count = len(frame_buffer) - 1
                        for _ in range(dropped_count):
                            record_dropped(metrics)
                        payload, received_at = frame_buffer[-1]
                        frame_buffer.clear()
                    else:
                        payload, received_at = frame_buffer.pop(0)

                    if frame_buffer:
                        frame_ready.set()
                    else:
                        frame_ready.clear()

            if payload is None:
                if stop_event.is_set():
                    break
                continue

            result = await worker_pool.process_frame(session_id, payload)
            worker_latency_ms = float(result.get("latency_ms", 0.0))
            total_latency_ms = (
                (loop.time() - received_at) * 1000 if received_at is not None else worker_latency_ms
            )

            record_processed(
                metrics,
                output_bytes=len(result["frame_data"]),
                worker_latency_ms=worker_latency_ms,
                total_latency_ms=total_latency_ms,
                stage_metrics=result.get("metrics", {}),
                worker_id=result.get("worker_id"),
            )
            await send_bytes_safe(result["frame_data"])

            if stop_event.is_set():
                async with frame_lock:
                    if not frame_buffer:
                        break

    try:
        await send_json_safe({
            "type": "connected",
            "session_id": session_id,
            "worker_id": session.get("worker_id"),
            "config": session["config"],
        })

        receiver_task = asyncio.create_task(receiver_loop())
        processor_task = asyncio.create_task(processor_loop())

        await receiver_task
        stop_event.set()
        frame_ready.set()
        await processor_task

    except WebSocketDisconnect:
        logger.info(f"Real-time session disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Real-time processing error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            logger.debug("Could not send error message to closed websocket")
    finally:
        stop_event.set()
        frame_ready.set()
        for task in (receiver_task, processor_task):
            if task is not None and not task.done():
                task.cancel()
        session["status"] = "closed"
        frame_count = int(metrics.get("processed_frames", 0))
        avg_latency = float(metrics.get("avg_total_latency_ms", 0.0))
        logger.info(
            f"Real-time session ended: {session_id} "
            f"(frames: {frame_count}, avg latency: {avg_latency:.1f}ms)"
        )


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
        logger.info("Status WebSocket disconnected")
    finally:
        manager.disconnect(websocket, "status")
