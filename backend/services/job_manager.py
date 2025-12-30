"""
Job management service for tracking generation jobs.
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
import uuid
from loguru import logger

from backend.core.models import (
    JobStatus,
    JobCreate,
    JobResponse,
    JobProgress,
    GenerateRequest,
)
from backend.core.exceptions import JobNotFoundError, JobCancelledError


@dataclass
class Job:
    """Internal job representation."""

    id: str
    request: GenerateRequest
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_step: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal tracking
    celery_task_id: Optional[str] = None
    cancelled: bool = False

    def to_response(self) -> JobResponse:
        """Convert to API response model."""
        return JobResponse(
            id=self.id,
            status=self.status,
            progress=self.progress,
            current_step=self.current_step,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            request=self.request,
            result_url=self.result_url,
            thumbnail_url=self.thumbnail_url,
            error=self.error,
            metadata=self.metadata,
        )

    def to_progress(self) -> JobProgress:
        """Convert to progress update model."""
        return JobProgress(
            job_id=self.id,
            status=self.status,
            progress=self.progress,
            current_step=self.current_step,
            eta_seconds=self._estimate_eta(),
            preview_url=self.thumbnail_url,
            error=self.error,
        )

    def _estimate_eta(self) -> Optional[float]:
        """Estimate remaining time based on progress."""
        if self.progress <= 0 or self.started_at is None:
            return None

        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        if self.progress >= 100:
            return 0

        # Simple linear estimation
        total_estimated = elapsed / (self.progress / 100)
        remaining = total_estimated - elapsed
        return max(0, remaining)


# Type for progress callbacks
ProgressCallback = Callable[[JobProgress], None]


class JobManager:
    """
    Manages generation jobs.

    Features:
    - Job creation and tracking
    - Status updates and progress tracking
    - Job cancellation
    - Callback system for real-time updates
    """

    def __init__(self, max_concurrent_jobs: int = None):
        self.max_concurrent_jobs = max_concurrent_jobs or 3

        # Job storage (in production, use a database)
        self._jobs: Dict[str, Job] = {}

        # Progress callbacks per job
        self._callbacks: Dict[str, List[ProgressCallback]] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def create_job(self, request: GenerateRequest) -> Job:
        """
        Create a new generation job.

        Args:
            request: The generation request

        Returns:
            Created Job object
        """
        async with self._lock:
            # Check concurrent job limit
            active_count = sum(
                1 for job in self._jobs.values()
                if job.status in [JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.GENERATING]
            )

            job_id = str(uuid.uuid4())
            job = Job(
                id=job_id,
                request=request,
                status=JobStatus.PENDING if active_count < self.max_concurrent_jobs else JobStatus.QUEUED,
            )

            self._jobs[job_id] = job
            logger.info(f"Created job: {job_id}")

            return job

    def get_job(self, job_id: str) -> Job:
        """
        Get a job by ID.

        Raises JobNotFoundError if not found.
        """
        if job_id not in self._jobs:
            raise JobNotFoundError(job_id)
        return self._jobs[job_id]

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Job:
        """
        Update job status.

        Args:
            job_id: Job ID to update
            status: New status
            progress: Optional progress percentage
            current_step: Optional current step description
            error: Optional error message

        Returns:
            Updated Job object
        """
        job = self.get_job(job_id)

        async with self._lock:
            job.status = status

            if progress is not None:
                job.progress = progress

            if current_step is not None:
                job.current_step = current_step

            if error is not None:
                job.error = error

            # Update timestamps
            if status == JobStatus.PROCESSING and job.started_at is None:
                job.started_at = datetime.utcnow()

            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_at = datetime.utcnow()

        # Notify callbacks
        await self._notify_progress(job)

        logger.debug(f"Job {job_id}: {status.value} ({progress:.1f}%)" if progress else f"Job {job_id}: {status.value}")

        return job

    async def update_progress(
        self,
        job_id: str,
        progress: float,
        current_step: str = "",
    ) -> Job:
        """
        Update job progress.

        Args:
            job_id: Job ID to update
            progress: Progress percentage (0-100)
            current_step: Current step description

        Returns:
            Updated Job object
        """
        job = self.get_job(job_id)

        # Check if cancelled
        if job.cancelled:
            raise JobCancelledError(job_id)

        async with self._lock:
            job.progress = progress
            job.current_step = current_step

            # Auto-update status based on progress
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.utcnow()

        # Notify callbacks
        await self._notify_progress(job)

        return job

    async def complete_job(
        self,
        job_id: str,
        result_url: str,
        thumbnail_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """
        Mark a job as completed.

        Args:
            job_id: Job ID
            result_url: URL to the result
            thumbnail_url: Optional thumbnail URL
            metadata: Optional additional metadata

        Returns:
            Updated Job object
        """
        job = self.get_job(job_id)

        async with self._lock:
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.current_step = "Complete"
            job.result_url = result_url
            job.thumbnail_url = thumbnail_url
            job.completed_at = datetime.utcnow()

            if metadata:
                job.metadata.update(metadata)

        # Notify callbacks
        await self._notify_progress(job)

        logger.info(f"Job completed: {job_id}")

        # Start next queued job
        await self._start_next_queued()

        return job

    async def fail_job(
        self,
        job_id: str,
        error: str,
    ) -> Job:
        """
        Mark a job as failed.

        Args:
            job_id: Job ID
            error: Error message

        Returns:
            Updated Job object
        """
        job = self.get_job(job_id)

        async with self._lock:
            job.status = JobStatus.FAILED
            job.error = error
            job.completed_at = datetime.utcnow()

        # Notify callbacks
        await self._notify_progress(job)

        logger.error(f"Job failed: {job_id} - {error}")

        # Start next queued job
        await self._start_next_queued()

        return job

    async def cancel_job(self, job_id: str) -> Job:
        """
        Cancel a job.

        Args:
            job_id: Job ID to cancel

        Returns:
            Updated Job object
        """
        job = self.get_job(job_id)

        async with self._lock:
            job.cancelled = True
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            job.current_step = "Cancelled"

        # Notify callbacks
        await self._notify_progress(job)

        logger.info(f"Job cancelled: {job_id}")

        return job

    def is_cancelled(self, job_id: str) -> bool:
        """Check if a job is cancelled."""
        try:
            job = self.get_job(job_id)
            return job.cancelled
        except JobNotFoundError:
            return True

    async def _start_next_queued(self) -> None:
        """Start the next queued job if possible."""
        active_count = sum(
            1 for job in self._jobs.values()
            if job.status in [JobStatus.PROCESSING, JobStatus.GENERATING]
        )

        if active_count < self.max_concurrent_jobs:
            # Find oldest queued job
            queued = [
                job for job in self._jobs.values()
                if job.status == JobStatus.QUEUED
            ]
            if queued:
                queued.sort(key=lambda j: j.created_at)
                next_job = queued[0]
                next_job.status = JobStatus.PENDING
                logger.debug(f"Promoted queued job: {next_job.id}")

    def register_callback(
        self,
        job_id: str,
        callback: ProgressCallback,
    ) -> None:
        """
        Register a callback for job progress updates.

        Args:
            job_id: Job ID to monitor
            callback: Function to call on updates
        """
        if job_id not in self._callbacks:
            self._callbacks[job_id] = []
        self._callbacks[job_id].append(callback)

    def unregister_callback(
        self,
        job_id: str,
        callback: ProgressCallback,
    ) -> None:
        """Unregister a progress callback."""
        if job_id in self._callbacks:
            try:
                self._callbacks[job_id].remove(callback)
            except ValueError:
                pass

    async def _notify_progress(self, job: Job) -> None:
        """Notify all registered callbacks of progress update."""
        if job.id not in self._callbacks:
            return

        progress = job.to_progress()
        for callback in self._callbacks[job.id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                logger.warning(f"Callback error for job {job.id}: {e}")

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> List[Job]:
        """
        List jobs.

        Args:
            status: Optional status filter
            limit: Maximum jobs to return

        Returns:
            List of Job objects
        """
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = sum(
                1 for job in self._jobs.values() if job.status == status
            )

        return {
            "total": len(self._jobs),
            "by_status": status_counts,
            "active": status_counts.get("processing", 0) + status_counts.get("generating", 0),
            "queued": status_counts.get("queued", 0),
        }

    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed/failed jobs.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of jobs removed
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = []

        for job_id, job in self._jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.completed_at and job.completed_at < cutoff:
                    to_remove.append(job_id)

        async with self._lock:
            for job_id in to_remove:
                del self._jobs[job_id]
                if job_id in self._callbacks:
                    del self._callbacks[job_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")

        return len(to_remove)


# Singleton instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
