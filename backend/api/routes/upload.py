"""
File upload endpoints.
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from backend.core.models import UploadResponse
from backend.core.exceptions import InvalidVideoError, InvalidImageError
from backend.services.file_manager import get_file_manager


router = APIRouter()


@router.post("/upload/image", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(..., description="Reference image file"),
):
    """
    Upload a reference image.

    Supported formats: JPEG, PNG, WebP, BMP

    The image should contain a clear view of the person/character
    you want to use as the reference for video generation.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate content type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {content_type}. Expected image/*"
        )

    try:
        # Read file content
        content = await file.read()

        # Save using file manager
        file_manager = get_file_manager()
        file_info = await file_manager.save_upload(
            content=content,
            filename=file.filename,
            generate_thumbnail=True,
        )

        logger.info(f"Uploaded image: {file_info.id} ({file.filename})")

        return file_manager.to_upload_response(file_info)

    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.post("/upload/video", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(..., description="Input video file"),
):
    """
    Upload an input video for processing.

    Supported formats: MP4, MOV, MPEG, AVI, WebM, MKV

    The video should contain the motion/poses you want to transfer
    to the reference character.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate content type
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {content_type}. Expected video/*"
        )

    try:
        # Read file content
        content = await file.read()

        # Save using file manager
        file_manager = get_file_manager()
        file_info = await file_manager.save_upload(
            content=content,
            filename=file.filename,
            generate_thumbnail=True,
        )

        logger.info(
            f"Uploaded video: {file_info.id} ({file.filename}, "
            f"{file_info.metadata.get('duration', 0):.1f}s)"
        )

        return file_manager.to_upload_response(file_info)

    except InvalidVideoError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.post("/upload/batch", response_model=List[UploadResponse])
async def upload_batch(
    files: List[UploadFile] = File(..., description="Multiple files to upload"),
):
    """
    Upload multiple files at once.

    Accepts both images and videos. Each file is processed individually.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    file_manager = get_file_manager()
    results = []
    errors = []

    for file in files:
        if not file.filename:
            continue

        try:
            content = await file.read()
            file_info = await file_manager.save_upload(
                content=content,
                filename=file.filename,
                generate_thumbnail=True,
            )
            results.append(file_manager.to_upload_response(file_info))

        except (InvalidVideoError, InvalidImageError) as e:
            errors.append({"filename": file.filename, "error": str(e)})
        except Exception as e:
            errors.append({"filename": file.filename, "error": "Upload failed"})

    if errors and not results:
        raise HTTPException(status_code=400, detail={"errors": errors})

    logger.info(f"Batch upload: {len(results)} succeeded, {len(errors)} failed")

    return results
