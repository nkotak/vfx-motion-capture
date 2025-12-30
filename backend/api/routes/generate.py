"""
Video generation endpoints.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from backend.core.models import (
    GenerateRequest,
    JobResponse,
    GenerationMode,
    QualityPreset,
)
from backend.core.exceptions import FileNotFoundError, InvalidInputError
from backend.services.file_manager import get_file_manager
from backend.services.job_manager import get_job_manager
from backend.services.prompt_parser import get_prompt_parser


router = APIRouter()


@router.post("/generate", response_model=JobResponse)
async def generate_video(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a video generation job.

    This endpoint queues a video generation job based on the provided
    reference image, input video, and prompt. The generation happens
    asynchronously - use the returned job_id to track progress.

    ## Modes

    - **auto**: Automatically selects the best mode based on prompt
    - **vace_pose_transfer**: Transfer poses from video to reference character
    - **vace_motion_transfer**: Transfer motion sequence to character
    - **wan_r2v**: Generate new video with reference character
    - **liveportrait**: Animate portrait with expressions
    - **deep_live_cam**: Real-time face swap

    ## Example

    ```json
    {
        "reference_image_id": "abc123",
        "input_video_id": "def456",
        "prompt": "Replace the person in the video with the reference character",
        "mode": "auto",
        "quality": "standard"
    }
    ```
    """
    file_manager = get_file_manager()
    job_manager = get_job_manager()
    prompt_parser = get_prompt_parser()

    # Validate reference image exists
    try:
        ref_image = file_manager.get_file(request.reference_image_id)
        if ref_image.file_type != "image":
            raise InvalidInputError(
                "Reference must be an image",
                field="reference_image_id"
            )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Reference image not found: {request.reference_image_id}"
        )

    # Validate input video if provided
    if request.input_video_id:
        try:
            input_video = file_manager.get_file(request.input_video_id)
            if input_video.file_type != "video":
                raise InvalidInputError(
                    "Input must be a video",
                    field="input_video_id"
                )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Input video not found: {request.input_video_id}"
            )

    # Parse prompt if in auto mode
    if request.mode == GenerationMode.AUTO:
        parsed = prompt_parser.parse(request.prompt)
        request.mode = parsed.mode

        # Merge parsed parameters
        if parsed.parameters:
            for key, value in parsed.parameters.items():
                if key not in request.extra_params:
                    request.extra_params[key] = value

        logger.debug(f"Auto mode selected: {request.mode} (confidence: {parsed.confidence:.2f})")

    # Create job
    job = await job_manager.create_job(request)

    # Queue background task
    background_tasks.add_task(run_generation, job.id)

    logger.info(f"Generation job created: {job.id} (mode: {request.mode})")

    return job.to_response()


async def run_generation(job_id: str):
    """Background task to run video generation."""
    from backend.workers.video_tasks import process_video_generation

    try:
        await process_video_generation(job_id)
    except Exception as e:
        logger.exception(f"Generation failed for job {job_id}: {e}")
        job_manager = get_job_manager()
        await job_manager.fail_job(job_id, str(e))


@router.post("/generate/preview", response_model=JobResponse)
async def generate_preview(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Generate a quick preview (first few frames only).

    Uses draft quality settings for faster generation.
    Useful for testing prompts and settings before full generation.
    """
    # Force draft quality for preview
    request.quality = QualityPreset.DRAFT
    request.duration = min(request.duration or 2.0, 2.0)  # Max 2 seconds
    request.extra_params["preview_mode"] = True

    return await generate_video(request, background_tasks)


@router.get("/generate/modes")
async def list_generation_modes():
    """
    List available generation modes with descriptions.
    """
    prompt_parser = get_prompt_parser()

    modes = []
    for mode in GenerationMode:
        modes.append({
            "value": mode.value,
            "name": mode.name,
            "description": prompt_parser.get_mode_description(mode),
            "suggested_prompt": prompt_parser.suggest_prompt(mode),
        })

    return {"modes": modes}


@router.get("/generate/quality-presets")
async def list_quality_presets():
    """
    List available quality presets.
    """
    presets = [
        {
            "value": QualityPreset.DRAFT.value,
            "name": "Draft",
            "description": "Fast preview quality, lower resolution",
            "estimated_time_factor": 0.25,
        },
        {
            "value": QualityPreset.STANDARD.value,
            "name": "Standard",
            "description": "Balanced quality and speed",
            "estimated_time_factor": 1.0,
        },
        {
            "value": QualityPreset.HIGH.value,
            "name": "High",
            "description": "High quality, slower generation",
            "estimated_time_factor": 2.0,
        },
        {
            "value": QualityPreset.ULTRA.value,
            "name": "Ultra",
            "description": "Maximum quality, longest generation time",
            "estimated_time_factor": 4.0,
        },
    ]

    return {"presets": presets}


@router.post("/generate/parse-prompt")
async def parse_prompt(prompt: str):
    """
    Parse a natural language prompt without starting generation.

    Useful for previewing how a prompt will be interpreted.
    """
    prompt_parser = get_prompt_parser()
    parsed = prompt_parser.parse(prompt)

    return {
        "mode": parsed.mode.value,
        "action": parsed.action,
        "subject": parsed.subject,
        "parameters": parsed.parameters,
        "confidence": parsed.confidence,
        "cleaned_prompt": parsed.cleaned_prompt,
        "mode_description": prompt_parser.get_mode_description(parsed.mode),
    }
