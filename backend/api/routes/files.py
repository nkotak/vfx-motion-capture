"""
File management endpoints.
"""

from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger

from backend.core.models import UploadResponse
from backend.core.exceptions import FileNotFoundError as VFXFileNotFoundError
from backend.services.file_manager import get_file_manager


router = APIRouter()


@router.get("/files", response_model=List[UploadResponse])
async def list_files(
    file_type: Optional[str] = Query(None, description="Filter by type (image/video)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum files to return"),
):
    """
    List uploaded files.

    Returns files sorted by upload time (newest first).
    """
    file_manager = get_file_manager()
    files = file_manager.list_files(file_type=file_type, limit=limit)
    return [file_manager.to_upload_response(f) for f in files]


@router.get("/files/{file_id}", response_model=UploadResponse)
async def get_file_info(file_id: str):
    """
    Get information about an uploaded file.
    """
    file_manager = get_file_manager()

    try:
        file_info = file_manager.get_file(file_id)
        return file_manager.to_upload_response(file_info)
    except VFXFileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")


@router.get("/files/{file_id}/download")
async def download_file(file_id: str):
    """
    Download an uploaded file.
    """
    file_manager = get_file_manager()

    try:
        file_info = file_manager.get_file(file_id)

        if not file_info.path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")

        return FileResponse(
            path=file_info.path,
            filename=file_info.original_filename,
            media_type="application/octet-stream",
        )

    except VFXFileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")


@router.get("/files/{file_id}/thumbnail")
async def get_thumbnail(file_id: str):
    """
    Get thumbnail for an uploaded file.
    """
    file_manager = get_file_manager()

    try:
        file_info = file_manager.get_file(file_id)

        if not file_info.thumbnail_path or not file_info.thumbnail_path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not available")

        return FileResponse(
            path=file_info.thumbnail_path,
            media_type="image/jpeg",
        )

    except VFXFileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete an uploaded file.
    """
    file_manager = get_file_manager()

    try:
        deleted = await file_manager.delete_file(file_id)

        if not deleted:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

        logger.info(f"File deleted: {file_id}")

        return {"status": "deleted", "file_id": file_id}

    except Exception as e:
        logger.exception(f"Failed to delete file {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")


@router.get("/outputs/{job_id}")
async def list_job_outputs(job_id: str):
    """
    List output files for a job.
    """
    file_manager = get_file_manager()
    output_dir = file_manager.get_output_path(job_id)

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"No outputs for job: {job_id}")

    files = []
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "url": f"/outputs/{job_id}/{file_path.name}",
            })

    return {"job_id": job_id, "files": files}


@router.get("/outputs/{job_id}/{filename}")
async def download_output(job_id: str, filename: str):
    """
    Download a specific output file.
    """
    file_manager = get_file_manager()
    file_path = file_manager.get_output_path(job_id, filename)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".gif": "image/gif",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
    )
