"""
Offline video generation tasks.
"""

import asyncio
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import imageio.v2 as imageio
from loguru import logger

from backend.core.config import settings
from backend.core.models import JobStatus, GenerationMode, QualityPreset
from backend.core.exceptions import JobCancelledError, VideoProcessingError


async def process_video_generation(job_id: str) -> None:
    """
    Main video generation task.
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.file_manager import get_file_manager
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

        if request.mode == GenerationMode.DEEP_LIVE_CAM:
            result = await generate_deep_live_cam(
                job_id=job_id,
                ref_image_path=ref_image_path,
                input_video_path=input_video_path,
                request=request,
            )
        elif request.mode == GenerationMode.LIVEPORTRAIT:
            result = await generate_liveportrait(
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
        elif request.mode == GenerationMode.VACE_POSE_TRANSFER:
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
        else:
            raise VideoProcessingError(f"Unsupported generation mode: {request.mode}")

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


async def _write_gif(input_path: Path, output_path: Path, fps: int) -> None:
    """Convert an intermediate video into an animated GIF."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise VideoProcessingError(f"Could not open intermediate video for GIF conversion: {input_path}")

    frames = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise VideoProcessingError("GIF conversion failed because no frames were available")
    imageio.mimsave(output_path, frames, fps=max(1, fps))


async def _materialize_final_output(
    raw_output: Path,
    final_output: Path,
    request: Any,
    video_processor: Any,
) -> None:
    """Convert or move a rendered MP4 into the requested output format."""
    final_output.parent.mkdir(parents=True, exist_ok=True)
    if final_output.exists():
        final_output.unlink()

    if request.output_format.value == "mp4":
        shutil.move(str(raw_output), str(final_output))
        return
    if request.output_format.value == "webm":
        await video_processor.convert_format(raw_output, final_output, codec="libvpx-vp9", crf=24)
        raw_output.unlink(missing_ok=True)
        return
    if request.output_format.value == "gif":
        await _write_gif(raw_output, final_output, request.fps)
        raw_output.unlink(missing_ok=True)
        return
    raise VideoProcessingError(f"Unsupported output format: {request.output_format.value}")


async def _finalize_generated_video(
    job_id: str,
    request: Any,
    raw_output: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate rendered output, create thumbnails, and convert to the requested format."""
    from backend.services.file_manager import get_file_manager
    from backend.services.video_processor import get_video_processor

    file_manager = get_file_manager()
    video_processor = get_video_processor()

    if not raw_output.exists() or raw_output.stat().st_size == 0:
        raise VideoProcessingError(f"Generated output missing or empty: {raw_output}")

    validated = await video_processor.get_metadata(raw_output)

    output_dir = file_manager.get_output_path(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = output_dir / "thumb.jpg"
    await video_processor.generate_thumbnail(raw_output, thumb_path)

    final_output = output_dir / f"output.{request.output_format.value}"
    await _materialize_final_output(raw_output, final_output, request, video_processor)

    result_metadata = {
        "duration": validated.duration,
        "fps": validated.fps,
        "resolution": (validated.width, validated.height),
        "mode": request.mode.value,
    }
    if metadata:
        result_metadata.update(metadata)

    return {
        "filename": final_output.name,
        "thumbnail": thumb_path.name,
        "metadata": result_metadata,
    }


def _temporary_output(job_id: str, stem: str) -> Path:
    return settings.temp_dir / f"{job_id}_{stem}.mp4"


async def generate_deep_live_cam(
    job_id: str,
    ref_image_path: Path,
    input_video_path: Optional[Path],
    request: Any,
) -> Dict[str, Any]:
    """
    Generate video using Face Swap directly.
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.inference.face_swapper import get_face_swapper

    job_manager = get_job_manager()
    face_swapper = get_face_swapper()

    if not input_video_path:
        raise VideoProcessingError("Input video required for face swap")

    # Load source image
    source_img = cv2.imread(str(ref_image_path))
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

    # Pre-cache the source face for efficient frame processing
    await face_swapper.set_source_face(source_img)

    await job_manager.update_progress(job_id, 10, "Processing video")

    # Process video frame by frame
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temporary output
    temp_output = _temporary_output(job_id, "swap")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))

    processed_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Swap face
            # Convert to RGB for processing if needed, but swapper returns BGR usually if using cv2
            # My swapper implementation assumed RGB input/output, checking...
            # The swapper uses detected faces which uses RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_rgb = await face_swapper.swap_face(source_img, frame_rgb)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

            out.write(result_bgr)

            processed_frames += 1
            if processed_frames % 10 == 0:
                progress = (processed_frames / total_frames) * 80 + 10
                await job_manager.update_progress(job_id, progress, f"Swapping frame {processed_frames}/{total_frames}")
                # Yield to event loop
                await asyncio.sleep(0)

    finally:
        cap.release()
        out.release()

    await job_manager.update_progress(job_id, 95, "Finalizing")
    return await _finalize_generated_video(job_id, request, temp_output)


async def generate_liveportrait(
    job_id: str,
    ref_image_path: Path,
    input_video_path: Optional[Path],
    request: Any,
) -> Dict[str, Any]:
    """
    Generate video using LivePortrait service.
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.inference.live_portrait import get_live_portrait_service

    job_manager = get_job_manager()
    lp_service = get_live_portrait_service()

    await job_manager.update_progress(job_id, 10, "Animating portrait")

    source_img = cv2.imread(str(ref_image_path))
    result_path = lp_service.process(source_img, str(input_video_path) if input_video_path else "")

    if not result_path:
        raise VideoProcessingError("LivePortrait generation failed")

    temp_output = _temporary_output(job_id, "liveportrait")
    if Path(result_path) != temp_output:
        shutil.copy(result_path, temp_output)

    await job_manager.update_progress(job_id, 95, "Finalizing")
    return await _finalize_generated_video(
        job_id,
        request,
        temp_output,
        metadata={"implementation": "builtin_landmark_renderer"},
    )


async def generate_wan_r2v(
    job_id: str,
    ref_image_path: Path,
    request: Any,
) -> Dict[str, Any]:
    """
    Generate video using Wan Video service.
    """
    from backend.services.job_manager import get_job_manager
    from backend.services.inference.wan_video import get_wan_video_service

    job_manager = get_job_manager()
    wan_service = get_wan_video_service()
    reference = cv2.imread(str(ref_image_path), cv2.IMREAD_UNCHANGED)
    if reference is None:
        raise VideoProcessingError(f"Could not load reference image: {ref_image_path}")

    await job_manager.update_progress(job_id, 20, "Generating video")

    temp_output = _temporary_output(job_id, "wan")
    result = await wan_service.generate(
        prompt=request.extra_params.get("cleaned_prompt", request.prompt),
        reference_image=reference,
        output_path=temp_output,
        request=request,
    )

    await job_manager.update_progress(job_id, 95, "Finalizing")
    return await _finalize_generated_video(job_id, request, Path(result["output_path"]), result.get("metadata"))


async def generate_vace_pose_transfer(
    job_id: str,
    ref_image_path: Path,
    input_video_path: Optional[Path],
    request: Any,
) -> Dict[str, Any]:
    """Generate a pose transfer video with the dedicated VACE path."""
    from backend.services.job_manager import get_job_manager
    from backend.services.inference.vace_video import get_vace_video_service

    if not input_video_path:
        raise VideoProcessingError("Input video required for pose transfer")

    job_manager = get_job_manager()
    vace_service = get_vace_video_service()
    temp_output = _temporary_output(job_id, "vace_pose")

    await job_manager.update_status(job_id, JobStatus.EXTRACTING_POSE, progress=15, current_step="Extracting pose controls")
    result = await vace_service.generate_pose_transfer(
        reference_image_path=ref_image_path,
        input_video_path=input_video_path,
        output_path=temp_output,
        request=request,
        progress_callback=lambda ratio, step: job_manager.update_progress(job_id, 15 + ratio * 70, step),
    )
    await job_manager.update_status(job_id, JobStatus.GENERATING, progress=90, current_step="Finalizing pose transfer")
    return await _finalize_generated_video(job_id, request, Path(result["output_path"]), result.get("metadata"))


async def generate_vace_motion_transfer(
    job_id: str,
    ref_image_path: Path,
    input_video_path: Optional[Path],
    request: Any,
) -> Dict[str, Any]:
    """Generate a motion transfer video with the dedicated VACE path."""
    from backend.services.job_manager import get_job_manager
    from backend.services.inference.vace_video import get_vace_video_service

    if not input_video_path:
        raise VideoProcessingError("Input video required for motion transfer")

    job_manager = get_job_manager()
    vace_service = get_vace_video_service()
    temp_output = _temporary_output(job_id, "vace_motion")

    await job_manager.update_status(job_id, JobStatus.EXTRACTING_POSE, progress=15, current_step="Extracting motion controls")
    result = await vace_service.generate_motion_transfer(
        reference_image_path=ref_image_path,
        input_video_path=input_video_path,
        output_path=temp_output,
        request=request,
        progress_callback=lambda ratio, step: job_manager.update_progress(job_id, 15 + ratio * 70, step),
    )
    await job_manager.update_status(job_id, JobStatus.GENERATING, progress=90, current_step="Finalizing motion transfer")
    return await _finalize_generated_video(job_id, request, Path(result["output_path"]), result.get("metadata"))

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

async def cleanup_expired():
    """Periodic task to clean up expired files and jobs."""
    from backend.services.file_manager import get_file_manager
    from backend.services.job_manager import get_job_manager

    file_manager = get_file_manager()
    job_manager = get_job_manager()

    files_cleaned = await file_manager.cleanup_expired()
    jobs_cleaned = await job_manager.cleanup_old_jobs()

    logger.info(f"Cleanup: {files_cleaned} files, {jobs_cleaned} jobs")
