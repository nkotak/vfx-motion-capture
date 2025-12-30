"""
Celery tasks for video generation.
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger

from backend.core.config import settings
from backend.core.models import JobStatus, GenerationMode, QualityPreset
from backend.core.exceptions import JobCancelledError


async def process_video_generation(job_id: str) -> None:
    """
    Main video generation task.

    Handles the complete pipeline:
    1. Load inputs (reference image, input video)
    2. Extract poses/faces from input video
    3. Run AI generation via ComfyUI
    4. Post-process and encode output
    5. Save results
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.file_manager import get_file_manager
    from backend.services.video_processor import get_video_processor
    from backend.services.pose_extractor import get_pose_extractor
    from backend.services.face_detector import get_face_detector
    from backend.services.comfyui_client import get_comfyui_client

    job_manager = get_job_manager()
    file_manager = get_file_manager()

    try:
        # Get job
        job = job_manager.get_job(job_id)
        request = job.request

        # Update status to processing
        await job_manager.update_status(
            job_id,
            JobStatus.PROCESSING,
            progress=0,
            current_step="Loading inputs",
        )

        # Check cancellation
        if job_manager.is_cancelled(job_id):
            raise JobCancelledError(job_id)

        # Load reference image
        ref_file = file_manager.get_file(request.reference_image_id)
        ref_image_path = ref_file.path

        # Load input video if provided
        input_video_path = None
        if request.input_video_id:
            video_file = file_manager.get_file(request.input_video_id)
            input_video_path = video_file.path

        logger.info(f"Job {job_id}: Processing with mode {request.mode}")

        # Route to appropriate generator
        if request.mode == GenerationMode.VACE_POSE_TRANSFER:
            result = await generate_vace_pose_transfer(
                job_id=job_id,
                ref_image_path=ref_image_path,
                input_video_path=input_video_path,
                request=request,
            )

        elif request.mode == GenerationMode.VACE_MOTION_TRANSFER:
            result = await generate_vace_motion_transfer(
                job_id=job_id,
                ref_image_path=ref_image_path,
                input_video_path=input_video_path,
                request=request,
            )

        elif request.mode == GenerationMode.WAN_R2V:
            result = await generate_wan_r2v(
                job_id=job_id,
                ref_image_path=ref_image_path,
                request=request,
            )

        elif request.mode == GenerationMode.LIVEPORTRAIT:
            result = await generate_liveportrait(
                job_id=job_id,
                ref_image_path=ref_image_path,
                input_video_path=input_video_path,
                request=request,
            )

        elif request.mode == GenerationMode.DEEP_LIVE_CAM:
            result = await generate_deep_live_cam(
                job_id=job_id,
                ref_image_path=ref_image_path,
                input_video_path=input_video_path,
                request=request,
            )

        else:
            # Default to VACE pose transfer
            result = await generate_vace_pose_transfer(
                job_id=job_id,
                ref_image_path=ref_image_path,
                input_video_path=input_video_path,
                request=request,
            )

        # Complete job
        await job_manager.complete_job(
            job_id,
            result_url=f"/outputs/{job_id}/{result['filename']}",
            thumbnail_url=f"/outputs/{job_id}/{result.get('thumbnail', 'thumb.jpg')}",
            metadata=result.get("metadata", {}),
        )

        logger.info(f"Job {job_id}: Completed successfully")

    except JobCancelledError:
        logger.info(f"Job {job_id}: Cancelled")
        await job_manager.update_status(job_id, JobStatus.CANCELLED)

    except Exception as e:
        logger.exception(f"Job {job_id}: Failed with error: {e}")
        await job_manager.fail_job(job_id, str(e))
        raise


async def generate_vace_pose_transfer(
    job_id: str,
    ref_image_path: Path,
    input_video_path: Optional[Path],
    request: Any,
) -> Dict[str, Any]:
    """
    Generate video using Wan VACE pose transfer.

    Pipeline:
    1. Extract poses from input video
    2. Load VACE workflow
    3. Upload reference image and pose video to ComfyUI
    4. Execute workflow
    5. Download and process result
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.file_manager import get_file_manager
    from backend.services.video_processor import get_video_processor
    from backend.services.pose_extractor import get_pose_extractor
    from backend.services.comfyui_client import get_comfyui_client

    job_manager = get_job_manager()
    file_manager = get_file_manager()
    video_processor = get_video_processor()
    pose_extractor = get_pose_extractor()
    comfyui = get_comfyui_client()

    output_dir = file_manager.get_output_path(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract poses from input video (20%)
    await job_manager.update_progress(job_id, 5, "Extracting poses from video")

    if input_video_path:
        pose_sequence = await pose_extractor.extract_from_video(
            input_video_path,
            fps=request.fps,
            progress_callback=lambda p, s: asyncio.create_task(
                job_manager.update_progress(job_id, 5 + p * 0.15, s)
            ),
        )

        # Render pose video
        pose_video_path = settings.temp_dir / f"{job_id}_poses.mp4"
        await pose_extractor.render_pose_video(
            pose_sequence,
            pose_video_path,
            background="black",
        )
    else:
        pose_video_path = None

    # Step 2: Load and prepare workflow (25%)
    await job_manager.update_progress(job_id, 20, "Preparing workflow")

    workflow_path = Path(__file__).parent.parent / "comfyui_workflows" / "wan_vace_pose_transfer.json"
    workflow = await comfyui.load_workflow(workflow_path)

    # Step 3: Upload files to ComfyUI (30%)
    await job_manager.update_progress(job_id, 25, "Uploading files to ComfyUI")

    ref_upload = await comfyui.upload_image(ref_image_path)

    if pose_video_path:
        pose_upload = await comfyui.upload_video(pose_video_path)
    else:
        pose_upload = None

    # Step 4: Configure workflow parameters (35%)
    await job_manager.update_progress(job_id, 30, "Configuring generation parameters")

    # Get quality settings
    quality_settings = get_quality_settings(request.quality)

    parameters = {
        "reference_image": ref_upload["name"],
        "control_video": pose_upload["name"] if pose_upload else "",
        "prompt": request.prompt,
        "negative_prompt": "blurry, distorted, low quality, artifacts",
        "strength": request.strength,
        "steps": quality_settings["steps"],
        "cfg_scale": quality_settings["cfg_scale"],
        "width": request.resolution[0] if request.resolution else 640,
        "height": request.resolution[1] if request.resolution else 640,
        "frames": int((request.duration or 5) * request.fps),
        "fps": request.fps,
        "seed": request.seed or -1,
    }

    workflow = comfyui.inject_parameters(workflow, parameters)

    # Step 5: Execute workflow (35% - 90%)
    await job_manager.update_progress(job_id, 35, "Generating video")

    def progress_callback(progress):
        pct = 35 + (progress.progress / progress.max_progress) * 55
        asyncio.create_task(
            job_manager.update_progress(job_id, pct, progress.current_step)
        )

    result = await comfyui.execute_workflow(
        workflow,
        progress_callback=progress_callback,
    )

    if not result.success:
        raise Exception(f"ComfyUI execution failed: {result.error}")

    # Step 6: Post-process output (90% - 100%)
    await job_manager.update_progress(job_id, 90, "Processing output")

    # Copy output video to job directory
    if result.videos:
        output_video = result.videos[0]
        final_output = output_dir / f"output.{request.output_format.value}"

        if output_video.suffix.lower() != f".{request.output_format.value}":
            await video_processor.convert_format(output_video, final_output)
        else:
            import shutil
            shutil.copy(output_video, final_output)

        # Generate thumbnail
        thumb_path = output_dir / "thumb.jpg"
        await video_processor.generate_thumbnail(final_output, thumb_path)

        # Get output metadata
        metadata = await video_processor.get_metadata(final_output)

        return {
            "filename": final_output.name,
            "thumbnail": thumb_path.name,
            "metadata": {
                "duration": metadata.duration,
                "fps": metadata.fps,
                "resolution": (metadata.width, metadata.height),
                "mode": request.mode.value,
            },
        }

    raise Exception("No output video generated")


async def generate_vace_motion_transfer(
    job_id: str,
    ref_image_path: Path,
    input_video_path: Optional[Path],
    request: Any,
) -> Dict[str, Any]:
    """Generate video using Wan VACE motion transfer."""
    # Similar to pose transfer but uses motion control instead
    return await generate_vace_pose_transfer(
        job_id, ref_image_path, input_video_path, request
    )


async def generate_wan_r2v(
    job_id: str,
    ref_image_path: Path,
    request: Any,
) -> Dict[str, Any]:
    """
    Generate video using Wan 2.6 Reference-to-Video.

    Creates a new video featuring the reference character.
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.file_manager import get_file_manager
    from backend.services.video_processor import get_video_processor
    from backend.services.comfyui_client import get_comfyui_client

    job_manager = get_job_manager()
    file_manager = get_file_manager()
    video_processor = get_video_processor()
    comfyui = get_comfyui_client()

    output_dir = file_manager.get_output_path(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load workflow
    await job_manager.update_progress(job_id, 10, "Loading Wan R2V workflow")

    workflow_path = Path(__file__).parent.parent / "comfyui_workflows" / "wan_r2v_character.json"
    workflow = await comfyui.load_workflow(workflow_path)

    # Upload reference
    await job_manager.update_progress(job_id, 20, "Uploading reference image")
    ref_upload = await comfyui.upload_image(ref_image_path)

    # Configure parameters
    quality_settings = get_quality_settings(request.quality)

    parameters = {
        "reference_image": ref_upload["name"],
        "prompt": request.prompt,
        "negative_prompt": "blurry, distorted, low quality",
        "duration": request.duration or 5,
        "fps": request.fps,
        "steps": quality_settings["steps"],
        "cfg_scale": quality_settings["cfg_scale"],
        "seed": request.seed or -1,
    }

    workflow = comfyui.inject_parameters(workflow, parameters)

    # Execute
    await job_manager.update_progress(job_id, 30, "Generating with Wan R2V")

    result = await comfyui.execute_workflow(
        workflow,
        progress_callback=lambda p: asyncio.create_task(
            job_manager.update_progress(job_id, 30 + (p.progress / p.max_progress) * 60, p.current_step)
        ),
    )

    if not result.success:
        raise Exception(f"Generation failed: {result.error}")

    # Process output
    await job_manager.update_progress(job_id, 90, "Finalizing output")

    if result.videos:
        final_output = output_dir / f"output.{request.output_format.value}"
        import shutil
        shutil.copy(result.videos[0], final_output)

        thumb_path = output_dir / "thumb.jpg"
        await video_processor.generate_thumbnail(final_output, thumb_path)

        metadata = await video_processor.get_metadata(final_output)

        return {
            "filename": final_output.name,
            "thumbnail": thumb_path.name,
            "metadata": {
                "duration": metadata.duration,
                "fps": metadata.fps,
                "resolution": (metadata.width, metadata.height),
                "mode": request.mode.value,
            },
        }

    raise Exception("No output generated")


async def generate_liveportrait(
    job_id: str,
    ref_image_path: Path,
    input_video_path: Optional[Path],
    request: Any,
) -> Dict[str, Any]:
    """
    Generate video using LivePortrait.

    Animates a portrait image using expressions from a driving video.
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.file_manager import get_file_manager
    from backend.services.video_processor import get_video_processor
    from backend.services.comfyui_client import get_comfyui_client

    job_manager = get_job_manager()
    file_manager = get_file_manager()
    video_processor = get_video_processor()
    comfyui = get_comfyui_client()

    output_dir = file_manager.get_output_path(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    await job_manager.update_progress(job_id, 10, "Loading LivePortrait workflow")

    workflow_path = Path(__file__).parent.parent / "comfyui_workflows" / "liveportrait_animate.json"
    workflow = await comfyui.load_workflow(workflow_path)

    # Upload files
    await job_manager.update_progress(job_id, 20, "Uploading files")
    ref_upload = await comfyui.upload_image(ref_image_path)

    video_upload = None
    if input_video_path:
        video_upload = await comfyui.upload_video(input_video_path)

    # Configure
    parameters = {
        "source_image": ref_upload["name"],
        "driving_video": video_upload["name"] if video_upload else "",
        "relative_motion_mode": "source_video_smoothed",
        "smoothing": request.extra_params.get("smoothing", 0.5),
    }

    workflow = comfyui.inject_parameters(workflow, parameters)

    # Execute
    await job_manager.update_progress(job_id, 30, "Animating portrait")

    result = await comfyui.execute_workflow(
        workflow,
        progress_callback=lambda p: asyncio.create_task(
            job_manager.update_progress(job_id, 30 + (p.progress / p.max_progress) * 60, p.current_step)
        ),
    )

    if not result.success:
        raise Exception(f"LivePortrait failed: {result.error}")

    # Process output
    await job_manager.update_progress(job_id, 90, "Finalizing")

    if result.videos:
        final_output = output_dir / f"output.{request.output_format.value}"
        import shutil
        shutil.copy(result.videos[0], final_output)

        thumb_path = output_dir / "thumb.jpg"
        await video_processor.generate_thumbnail(final_output, thumb_path)

        metadata = await video_processor.get_metadata(final_output)

        return {
            "filename": final_output.name,
            "thumbnail": thumb_path.name,
            "metadata": {
                "duration": metadata.duration,
                "mode": request.mode.value,
            },
        }

    raise Exception("No output generated")


async def generate_deep_live_cam(
    job_id: str,
    ref_image_path: Path,
    input_video_path: Optional[Path],
    request: Any,
) -> Dict[str, Any]:
    """
    Generate video using Deep Live Cam face swap.
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.file_manager import get_file_manager
    from backend.services.video_processor import get_video_processor
    from backend.services.face_detector import get_face_detector
    from backend.services.comfyui_client import get_comfyui_client

    job_manager = get_job_manager()
    file_manager = get_file_manager()
    video_processor = get_video_processor()
    comfyui = get_comfyui_client()

    output_dir = file_manager.get_output_path(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    await job_manager.update_progress(job_id, 10, "Loading face swap workflow")

    workflow_path = Path(__file__).parent.parent / "comfyui_workflows" / "deep_live_cam.json"
    workflow = await comfyui.load_workflow(workflow_path)

    # Upload files
    await job_manager.update_progress(job_id, 20, "Uploading files")
    ref_upload = await comfyui.upload_image(ref_image_path)

    video_upload = None
    if input_video_path:
        video_upload = await comfyui.upload_video(input_video_path)

    # Configure
    parameters = {
        "source_face": ref_upload["name"],
        "target_video": video_upload["name"] if video_upload else "",
        "enhance_face": request.extra_params.get("enhance_face", True),
    }

    workflow = comfyui.inject_parameters(workflow, parameters)

    # Execute
    await job_manager.update_progress(job_id, 30, "Swapping faces")

    result = await comfyui.execute_workflow(
        workflow,
        progress_callback=lambda p: asyncio.create_task(
            job_manager.update_progress(job_id, 30 + (p.progress / p.max_progress) * 60, p.current_step)
        ),
    )

    if not result.success:
        raise Exception(f"Face swap failed: {result.error}")

    # Process output
    await job_manager.update_progress(job_id, 90, "Finalizing")

    if result.videos:
        final_output = output_dir / f"output.{request.output_format.value}"
        import shutil
        shutil.copy(result.videos[0], final_output)

        thumb_path = output_dir / "thumb.jpg"
        await video_processor.generate_thumbnail(final_output, thumb_path)

        metadata = await video_processor.get_metadata(final_output)

        return {
            "filename": final_output.name,
            "thumbnail": thumb_path.name,
            "metadata": {
                "duration": metadata.duration,
                "mode": request.mode.value,
            },
        }

    raise Exception("No output generated")


def get_quality_settings(quality: QualityPreset) -> Dict[str, Any]:
    """Get generation settings based on quality preset."""
    settings = {
        QualityPreset.DRAFT: {
            "steps": 15,
            "cfg_scale": 5.0,
            "resolution_scale": 0.5,
        },
        QualityPreset.STANDARD: {
            "steps": 25,
            "cfg_scale": 7.0,
            "resolution_scale": 1.0,
        },
        QualityPreset.HIGH: {
            "steps": 40,
            "cfg_scale": 7.5,
            "resolution_scale": 1.0,
        },
        QualityPreset.ULTRA: {
            "steps": 60,
            "cfg_scale": 8.0,
            "resolution_scale": 1.5,
        },
    }
    return settings.get(quality, settings[QualityPreset.STANDARD])


# Periodic cleanup task
async def cleanup_expired():
    """Periodic task to clean up expired files and jobs."""
    from backend.services.file_manager import get_file_manager
    from backend.services.job_manager import get_job_manager

    file_manager = get_file_manager()
    job_manager = get_job_manager()

    files_cleaned = await file_manager.cleanup_expired()
    jobs_cleaned = await job_manager.cleanup_old_jobs()

    logger.info(f"Cleanup: {files_cleaned} files, {jobs_cleaned} jobs")
