"""
Job management endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from backend.core.models import JobStatus, JobResponse, JobProgress
from backend.core.exceptions import JobNotFoundError
from backend.services.job_manager import get_job_manager


router = APIRouter()


@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum jobs to return"),
):
    """
    List all generation jobs.

    Optionally filter by status. Returns jobs sorted by creation time (newest first).
    """
    job_manager = get_job_manager()
    jobs = job_manager.list_jobs(status=status, limit=limit)
    return [job.to_response() for job in jobs]


@router.get("/jobs/stats")
async def get_job_stats():
    """
    Get job statistics.

    Returns counts by status and other aggregate information.
    """
    job_manager = get_job_manager()
    return job_manager.get_stats()


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get details for a specific job.
    """
    job_manager = get_job_manager()

    try:
        job = job_manager.get_job(job_id)
        return job.to_response()
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("/jobs/{job_id}/progress", response_model=JobProgress)
async def get_job_progress(job_id: str):
    """
    Get current progress for a job.

    Returns lightweight progress information suitable for polling.
    """
    job_manager = get_job_manager()

    try:
        job = job_manager.get_job(job_id)
        return job.to_progress()
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str):
    """
    Cancel a running or queued job.

    Jobs that are already completed or failed cannot be cancelled.
    """
    job_manager = get_job_manager()

    try:
        job = job_manager.get_job(job_id)

        # Check if job can be cancelled
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job with status: {job.status.value}"
            )

        job = await job_manager.cancel_job(job_id)
        logger.info(f"Job cancelled: {job_id}")

        return job.to_response()

    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files.

    Only completed, failed, or cancelled jobs can be deleted.
    """
    job_manager = get_job_manager()

    try:
        job = job_manager.get_job(job_id)

        # Check if job can be deleted
        if job.status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.GENERATING]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete active job. Cancel it first."
            )

        # Delete output files
        from backend.services.file_manager import get_file_manager
        file_manager = get_file_manager()

        output_dir = file_manager.get_output_path(job_id)
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        # Remove from job manager
        # Note: In production, this would be a database delete
        job_manager._jobs.pop(job_id, None)

        logger.info(f"Job deleted: {job_id}")

        return {"status": "deleted", "job_id": job_id}

    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.post("/jobs/{job_id}/retry", response_model=JobResponse)
async def retry_job(job_id: str):
    """
    Retry a failed job.

    Creates a new job with the same parameters as the failed one.
    """
    job_manager = get_job_manager()

    try:
        original_job = job_manager.get_job(job_id)

        if original_job.status != JobStatus.FAILED:
            raise HTTPException(
                status_code=400,
                detail="Only failed jobs can be retried"
            )

        # Create new job with same request
        new_job = await job_manager.create_job(original_job.request)

        # Queue generation
        from fastapi import BackgroundTasks
        from backend.api.routes.generate import run_generation

        # Note: In a real app, we'd use the background tasks from the request context
        import asyncio
        asyncio.create_task(run_generation(new_job.id))

        logger.info(f"Retrying job {job_id} as {new_job.id}")

        return new_job.to_response()

    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the result video URL for a completed job.
    """
    job_manager = get_job_manager()

    try:
        job = job_manager.get_job(job_id)

        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Job not completed. Status: {job.status.value}"
            )

        if not job.result_url:
            raise HTTPException(
                status_code=404,
                detail="Result not available"
            )

        return {
            "job_id": job_id,
            "result_url": job.result_url,
            "thumbnail_url": job.thumbnail_url,
            "metadata": job.metadata,
        }

    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
