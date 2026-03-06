from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.core.models import GenerateRequest, GenerationMode, QualityPreset
from backend.services.inference.vace_video import VaceVideoService
from backend.services.inference.wan_video import WanVideoService
from backend.services.job_manager import JobManager
from backend.workers import video_tasks


def _write_reference_image(path: Path) -> None:
    image = np.zeros((240, 180, 3), dtype=np.uint8)
    image[:, :] = (60, 120, 220)
    cv2.circle(image, (90, 90), 40, (240, 210, 180), -1)
    cv2.rectangle(image, (55, 130), (125, 220), (35, 35, 35), -1)
    assert cv2.imwrite(str(path), image)


def _write_video(path: Path, *, frames: int = 12, size: tuple[int, int] = (160, 120), fps: int = 12) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            center_x = 30 + index * 8
            center_y = 50 + int(10 * np.sin(index / 2))
            cv2.circle(frame, (center_x, center_y), 14, (255, 220, 180), -1)
            cv2.rectangle(frame, (center_x - 10, center_y + 12), (center_x + 10, center_y + 45), (0, 200, 80), -1)
            cv2.line(frame, (center_x, center_y + 5), (center_x - 18, center_y + 25), (255, 255, 255), 3)
            cv2.line(frame, (center_x, center_y + 5), (center_x + 18, center_y + 25), (255, 255, 255), 3)
            writer.write(frame)
    finally:
        writer.release()


@pytest.mark.asyncio
async def test_wan_builtin_renderer_produces_valid_video(tmp_path):
    reference_path = tmp_path / "reference.png"
    output_path = tmp_path / "wan.mp4"
    _write_reference_image(reference_path)

    request = GenerateRequest(
        reference_image_id="reference",
        prompt="Generate a cinematic forest walk",
        mode=GenerationMode.WAN_R2V,
        quality=QualityPreset.STANDARD,
        fps=12,
        duration=2,
    )

    service = WanVideoService()
    reference = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
    result = await service.generate(request.prompt, reference, output_path, request)

    assert Path(result["output_path"]).exists()
    assert Path(result["output_path"]).stat().st_size > 0
    capture = cv2.VideoCapture(str(result["output_path"]))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        capture.release()


@pytest.mark.asyncio
async def test_vace_pose_transfer_renderer_produces_valid_video(tmp_path):
    reference_path = tmp_path / "reference.png"
    input_video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "pose.mp4"
    _write_reference_image(reference_path)
    _write_video(input_video_path)

    request = GenerateRequest(
        reference_image_id="reference",
        input_video_id="video",
        prompt="Transfer the motion to my character",
        mode=GenerationMode.VACE_POSE_TRANSFER,
        quality=QualityPreset.STANDARD,
        fps=12,
        duration=1,
        strength=0.9,
    )

    service = VaceVideoService()

    async def no_pose(*args, **kwargs):
        return []

    service.pose_extractor.extract_from_image = no_pose
    result = await service.generate_pose_transfer(reference_path, input_video_path, output_path, request)

    assert Path(result["output_path"]).exists()
    assert result["metadata"]["frames_rendered"] > 0
    capture = cv2.VideoCapture(str(result["output_path"]))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        capture.release()


@pytest.mark.asyncio
async def test_worker_routes_pose_transfer_without_fallback(monkeypatch, tmp_path):
    reference_path = tmp_path / "reference.png"
    input_video_path = tmp_path / "input.mp4"
    _write_reference_image(reference_path)
    _write_video(input_video_path)

    request = GenerateRequest(
        reference_image_id="reference",
        input_video_id="video",
        prompt="Replace the person in the video",
        mode=GenerationMode.VACE_POSE_TRANSFER,
        quality=QualityPreset.STANDARD,
        fps=12,
        duration=1,
    )

    job_manager = JobManager()
    job = await job_manager.create_job(request)
    file_manager = SimpleNamespace(
        get_file=lambda file_id: SimpleNamespace(
            file_type="image" if file_id == "reference" else "video",
            path=reference_path if file_id == "reference" else input_video_path,
        ),
        get_output_path=lambda job_id, filename=None: (
            (tmp_path / job_id / filename) if filename else (tmp_path / job_id)
        ),
    )

    called = {"pose": 0, "wan": 0}

    async def fake_pose_generator(**kwargs):
        called["pose"] += 1
        return {
            "filename": "output.mp4",
            "thumbnail": "thumb.jpg",
            "metadata": {"implementation": "test"},
        }

    async def fail_wan_generator(**kwargs):
        called["wan"] += 1
        raise AssertionError("Wan fallback should not be invoked for pose transfer")

    monkeypatch.setattr("backend.services.job_manager.get_job_manager", lambda: job_manager)
    monkeypatch.setattr("backend.services.file_manager.get_file_manager", lambda: file_manager)
    monkeypatch.setattr(video_tasks, "generate_vace_pose_transfer", fake_pose_generator)
    monkeypatch.setattr(video_tasks, "generate_wan_r2v", fail_wan_generator)

    await video_tasks.process_video_generation(job.id)

    completed = job_manager.get_job(job.id)
    assert completed.status.value == "completed"
    assert called["pose"] == 1
    assert called["wan"] == 0
