"""
Multiprocess worker pool for realtime JPEG frame processing.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger

from backend.core.config import settings
from backend.services.realtime.pipeline import (
    cleanup_processor,
    initialize_realtime_processor,
    process_frame,
)
from backend.services.realtime.shared_memory import (
    cleanup_shared_memory_payload,
    close_shared_memory,
    create_shared_memory_payload,
    read_shared_memory_payload,
)


def _worker_process_main(
    worker_id: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
) -> None:
    """Process entrypoint for realtime inference workers."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    processors: Dict[str, Dict[str, Any]] = {}
    sessions: Dict[str, Dict[str, Any]] = {}

    output_queue.put({
        "type": "worker_ready",
        "worker_id": worker_id,
    })

    try:
        while True:
            task = input_queue.get()
            task_type = task["type"]
            request_id = task.get("request_id")

            if task_type == "shutdown":
                if request_id:
                    output_queue.put({
                        "type": "shutdown_ack",
                        "request_id": request_id,
                        "worker_id": worker_id,
                    })
                break

            try:
                if task_type == "init_session":
                    session_id = task["session_id"]
                    session = task["session"]
                    processors[session_id] = loop.run_until_complete(
                        initialize_realtime_processor(session)
                    )
                    sessions[session_id] = session
                    output_queue.put({
                        "type": "session_ready",
                        "request_id": request_id,
                        "worker_id": worker_id,
                    })

                elif task_type == "process_frame":
                    session_id = task["session_id"]
                    if session_id not in processors:
                        raise KeyError(f"Unknown realtime session: {session_id}")

                    if task.get("frame_ref"):
                        frame_data = read_shared_memory_payload(task["frame_ref"], unlink=True)
                    else:
                        frame_data = task["frame_data"]

                    start_time = time.perf_counter()
                    result = loop.run_until_complete(
                        process_frame(
                            processors[session_id],
                            frame_data,
                            sessions[session_id],
                        )
                    )
                    frame_payload = result["frame_data"]
                    response_message = {
                        "type": "frame_result",
                        "request_id": request_id,
                        "worker_id": worker_id,
                        "metrics": result.get("metrics", {}),
                        "latency_ms": (time.perf_counter() - start_time) * 1000,
                        "transport_metrics": {
                            "output_shared_memory": False,
                            "output_bytes": len(frame_payload),
                        },
                    }
                    if (
                        settings.realtime_use_shared_memory
                        and len(frame_payload) >= settings.realtime_shared_memory_threshold_bytes
                    ):
                        frame_ref, shm = create_shared_memory_payload(frame_payload)
                        cleanup_required = False
                        try:
                            response_message["frame_ref"] = frame_ref
                            response_message["transport_metrics"]["output_shared_memory"] = True
                            output_queue.put(response_message)
                        except Exception:
                            cleanup_required = True
                            raise
                        finally:
                            close_shared_memory(shm, unlink=cleanup_required)
                        continue

                    output_queue.put({
                        **response_message,
                        "frame_data": frame_payload,
                    })

                elif task_type == "close_session":
                    session_id = task["session_id"]
                    processor = processors.pop(session_id, None)
                    sessions.pop(session_id, None)
                    if processor is not None:
                        loop.run_until_complete(cleanup_processor(processor))

                    output_queue.put({
                        "type": "session_closed",
                        "request_id": request_id,
                        "worker_id": worker_id,
                    })

            except Exception as exc:
                output_queue.put({
                    "type": "error",
                    "request_id": request_id,
                    "worker_id": worker_id,
                    "message": str(exc),
                })
    finally:
        for session_id, processor in list(processors.items()):
            try:
                loop.run_until_complete(cleanup_processor(processor))
            except Exception as exc:
                logger.warning(f"Worker {worker_id} cleanup failed for {session_id}: {exc}")

        output_queue.put(None)
        loop.close()


@dataclass
class _WorkerHandle:
    worker_id: int
    process: mp.Process
    input_queue: mp.Queue
    output_queue: mp.Queue
    result_thread: threading.Thread
    active_sessions: set[str] = field(default_factory=set)


class RealtimeWorkerPool:
    """Multiprocess realtime worker pool with per-session worker affinity."""

    def __init__(self, worker_processes: Optional[int] = None):
        self.worker_processes = max(1, worker_processes or settings.realtime_worker_processes)
        self._ctx = mp.get_context("spawn")
        self._workers: Dict[int, _WorkerHandle] = {}
        self._session_workers: Dict[str, int] = {}
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._pending_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._worker_stats: Dict[int, Dict[str, Any]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = False

    async def start(self) -> None:
        """Start worker processes and result listeners."""
        if self._started:
            return

        self._loop = asyncio.get_running_loop()

        for worker_id in range(self.worker_processes):
            input_queue: mp.Queue = self._ctx.Queue(maxsize=max(4, settings.realtime_buffer_size * 2))
            output_queue: mp.Queue = self._ctx.Queue(maxsize=max(4, settings.realtime_buffer_size * 2))
            process = self._ctx.Process(
                target=_worker_process_main,
                args=(worker_id, input_queue, output_queue),
                daemon=True,
            )
            process.start()

            result_thread = threading.Thread(
                target=self._result_listener,
                args=(worker_id, output_queue),
                daemon=True,
                name=f"realtime-worker-{worker_id}-listener",
            )
            result_thread.start()

            self._workers[worker_id] = _WorkerHandle(
                worker_id=worker_id,
                process=process,
                input_queue=input_queue,
                output_queue=output_queue,
                result_thread=result_thread,
            )
            self._worker_stats[worker_id] = {
                "worker_id": worker_id,
                "pending_requests": 0,
                "processed_requests": 0,
                "error_count": 0,
                "avg_latency_ms": 0.0,
                "last_latency_ms": 0.0,
                "shared_memory_in_count": 0,
                "shared_memory_in_bytes": 0,
                "shared_memory_out_count": 0,
                "shared_memory_out_bytes": 0,
                "inline_transport_in_count": 0,
                "inline_transport_in_bytes": 0,
                "inline_transport_out_count": 0,
                "inline_transport_out_bytes": 0,
                "input_queue_size": 0,
                "output_queue_size": 0,
            }

        self._started = True
        logger.info(f"Started realtime worker pool with {self.worker_processes} processes")

    async def shutdown(self) -> None:
        """Shut down workers and fail any pending requests."""
        if not self._started:
            return

        for worker_id in list(self._workers.keys()):
            try:
                await self._dispatch(worker_id, {"type": "shutdown"}, timeout=30.0)
            except Exception as exc:
                logger.warning(f"Failed to shut down realtime worker {worker_id}: {exc}")

        for worker in self._workers.values():
            worker.process.join(timeout=5.0)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=2.0)

        with self._pending_lock:
            pending = [entry["future"] for entry in self._pending.values()]
            self._pending.clear()

        for future in pending:
            if not future.done():
                future.cancel()

        self._workers.clear()
        self._session_workers.clear()
        self._worker_stats.clear()
        self._started = False
        logger.info("Realtime worker pool shut down")

    async def register_session(self, session_id: str, session: Dict[str, Any]) -> int:
        """Assign a session to a worker and warm its process-local models."""
        if not self._started:
            await self.start()

        if session_id in self._session_workers:
            return self._session_workers[session_id]

        worker = min(self._workers.values(), key=lambda item: len(item.active_sessions))
        await self._dispatch(
            worker.worker_id,
            {
                "type": "init_session",
                "session_id": session_id,
                "session": session,
            },
            timeout=180.0,
        )

        worker.active_sessions.add(session_id)
        self._session_workers[session_id] = worker.worker_id
        return worker.worker_id

    async def close_session(self, session_id: str) -> None:
        """Release a session from its assigned worker."""
        worker_id = self._session_workers.pop(session_id, None)
        if worker_id is None:
            return

        worker = self._workers.get(worker_id)
        if worker is None:
            return

        try:
            await self._dispatch(
                worker_id,
                {
                    "type": "close_session",
                    "session_id": session_id,
                },
                timeout=30.0,
            )
        finally:
            worker.active_sessions.discard(session_id)

    async def process_frame(self, session_id: str, frame_data: bytes) -> Dict[str, Any]:
        """Process a frame on the worker assigned to a session."""
        worker_id = self._session_workers.get(session_id)
        if worker_id is None:
            raise KeyError(f"Realtime session is not registered: {session_id}")

        task, shm = self._build_frame_task(session_id, frame_data)
        try:
            message = await self._dispatch(
                worker_id,
                task,
                timeout=max(30.0, settings.realtime_max_latency_ms / 1000 * 10),
            )
        finally:
            if shm is not None:
                close_shared_memory(shm)
        return {
            "frame_data": self._resolve_frame_payload(message),
            "latency_ms": message.get("latency_ms", 0.0),
            "metrics": message.get("metrics", {}),
            "transport_metrics": message.get("transport_metrics", {}),
            "worker_id": worker_id,
        }

    def _result_listener(self, worker_id: int, output_queue: mp.Queue) -> None:
        """Bridge worker results back into asyncio futures."""
        while True:
            message = output_queue.get()
            if message is None:
                return

            message_type = message.get("type")
            request_id = message.get("request_id")

            if request_id is None:
                if message_type == "worker_ready":
                    logger.info(f"Realtime worker {worker_id} ready")
                continue

            with self._pending_lock:
                pending_entry = self._pending.pop(request_id, None)

            if pending_entry is None:
                if message.get("frame_ref"):
                    cleanup_shared_memory_payload(message["frame_ref"])
                continue

            future = pending_entry["future"]
            if future.done() or self._loop is None:
                if message.get("frame_ref"):
                    cleanup_shared_memory_payload(message["frame_ref"])
                continue

            self._update_worker_stats(worker_id, pending_entry, message)

            if message_type == "error":
                error = RuntimeError(message.get("message", "Realtime worker error"))
                self._loop.call_soon_threadsafe(future.set_exception, error)
            else:
                self._loop.call_soon_threadsafe(future.set_result, message)

    async def _dispatch(
        self,
        worker_id: int,
        task: Dict[str, Any],
        *,
        timeout: float,
    ) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        task = dict(task)
        task["request_id"] = request_id

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        pending_entry = {
            "future": future,
            "worker_id": worker_id,
            "task_type": task["type"],
            "transport_metrics": task.get("transport_metrics", {}),
        }
        with self._pending_lock:
            self._pending[request_id] = pending_entry

        worker = self._workers[worker_id]
        try:
            await asyncio.to_thread(worker.input_queue.put, task)
            if task["type"] == "process_frame":
                with self._stats_lock:
                    stats = self._worker_stats[worker_id]
                    stats["pending_requests"] += 1
                    stats["input_queue_size"] = _safe_qsize(worker.input_queue)
            return await asyncio.wait_for(future, timeout=timeout)
        except Exception:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            if task["type"] == "process_frame":
                with self._stats_lock:
                    stats = self._worker_stats[worker_id]
                    if stats["pending_requests"] > 0:
                        stats["pending_requests"] -= 1
            raise

    def _resolve_frame_payload(self, message: Dict[str, Any]) -> bytes:
        """Resolve inline or shared-memory frame payloads."""
        if message.get("frame_ref"):
            return read_shared_memory_payload(message["frame_ref"], unlink=True)
        return message["frame_data"]

    def _build_frame_task(self, session_id: str, frame_data: bytes) -> tuple[Dict[str, Any], Any]:
        """Create a queue-friendly task payload for a frame."""
        task: Dict[str, Any] = {
            "type": "process_frame",
            "session_id": session_id,
        }
        shm = None
        transport_metrics = {
            "input_shared_memory": False,
            "input_bytes": len(frame_data),
        }
        if (
            settings.realtime_use_shared_memory
            and len(frame_data) >= settings.realtime_shared_memory_threshold_bytes
        ):
            frame_ref, shm = create_shared_memory_payload(frame_data)
            task["frame_ref"] = frame_ref
            transport_metrics["input_shared_memory"] = True
        else:
            task["frame_data"] = frame_data
        task["transport_metrics"] = transport_metrics
        return task, shm

    def _update_worker_stats(
        self,
        worker_id: int,
        pending_entry: Dict[str, Any],
        message: Dict[str, Any],
    ) -> None:
        """Update worker telemetry after a task result arrives."""
        with self._stats_lock:
            stats = self._worker_stats[worker_id]
            if stats["pending_requests"] > 0:
                stats["pending_requests"] -= 1
            stats["output_queue_size"] = _safe_qsize(self._workers[worker_id].output_queue)

            if pending_entry["task_type"] != "process_frame":
                if message.get("type") == "error":
                    stats["error_count"] += 1
                return

            transport_in = pending_entry.get("transport_metrics", {})
            if transport_in.get("input_shared_memory"):
                stats["shared_memory_in_count"] += 1
                stats["shared_memory_in_bytes"] += int(transport_in.get("input_bytes", 0))
            else:
                stats["inline_transport_in_count"] += 1
                stats["inline_transport_in_bytes"] += int(transport_in.get("input_bytes", 0))

            if message.get("type") == "error":
                stats["error_count"] += 1
                return

            stats["processed_requests"] += 1
            latency_ms = float(message.get("latency_ms", 0.0))
            processed_requests = stats["processed_requests"]
            stats["last_latency_ms"] = latency_ms
            if processed_requests <= 1:
                stats["avg_latency_ms"] = latency_ms
            else:
                previous = float(stats["avg_latency_ms"])
                stats["avg_latency_ms"] = previous + ((latency_ms - previous) / processed_requests)

            output_shared_memory = bool(message.get("frame_ref"))
            output_bytes = int(message.get("transport_metrics", {}).get("output_bytes", 0))
            if output_shared_memory:
                stats["shared_memory_out_count"] += 1
                stats["shared_memory_out_bytes"] += output_bytes
            else:
                stats["inline_transport_out_count"] += 1
                stats["inline_transport_out_bytes"] += output_bytes

    def snapshot_worker_stats(self) -> list[Dict[str, Any]]:
        """Return current worker telemetry for APIs and debug panels."""
        with self._stats_lock:
            snapshots: list[Dict[str, Any]] = []
            for worker_id, worker in self._workers.items():
                stats = dict(self._worker_stats.get(worker_id, {}))
                stats["active_sessions"] = len(worker.active_sessions)
                stats["session_ids"] = sorted(worker.active_sessions)
                stats["process_alive"] = worker.process.is_alive()
                stats["input_queue_size"] = _safe_qsize(worker.input_queue)
                stats["output_queue_size"] = _safe_qsize(worker.output_queue)
                queue_capacity = max(1, settings.realtime_buffer_size * 2)
                stats["saturation"] = min(
                    1.0,
                    float(stats.get("pending_requests", 0)) / float(queue_capacity),
                )
                snapshots.append(stats)
            return sorted(snapshots, key=lambda item: item["worker_id"])


def _safe_qsize(queue: mp.Queue) -> int:
    """Best-effort queue size for diagnostics."""
    try:
        return max(0, int(queue.qsize()))
    except (NotImplementedError, AttributeError):
        return -1


_realtime_worker_pool: Optional[RealtimeWorkerPool] = None


def get_realtime_worker_pool() -> RealtimeWorkerPool:
    """Get the process-local realtime worker pool singleton."""
    global _realtime_worker_pool
    if _realtime_worker_pool is None:
        _realtime_worker_pool = RealtimeWorkerPool()
    return _realtime_worker_pool


def reset_realtime_worker_pool() -> None:
    """Reset the realtime worker pool singleton."""
    global _realtime_worker_pool
    _realtime_worker_pool = None
