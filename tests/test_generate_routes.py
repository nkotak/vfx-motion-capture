from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend.api.routes import generate as generate_routes
from backend.core.exceptions import VFXException
from backend.core.models import GenerationMode, JobStatus, QualityPreset
from backend.services.job_manager import JobManager


class StubFileManager:
    def __init__(self):
        self.reference = SimpleNamespace(file_type="image", path="ref.png")
        self.video = SimpleNamespace(file_type="video", path="input.mp4")

    def get_file(self, file_id: str):
        if file_id == "reference":
            return self.reference
        if file_id == "video":
            return self.video
        raise generate_routes.FileNotFoundError(file_id)


@pytest.fixture()
def test_app(monkeypatch):
    file_manager = StubFileManager()
    job_manager = JobManager()

    async def noop_run_generation(job_id: str) -> None:
        return None

    monkeypatch.setattr(generate_routes, "get_file_manager", lambda: file_manager)
    monkeypatch.setattr(generate_routes, "get_job_manager", lambda: job_manager)
    monkeypatch.setattr(generate_routes, "run_generation", noop_run_generation)

    app = FastAPI()

    @app.exception_handler(VFXException)
    async def handle_vfx_exception(request, exc: VFXException):
        return JSONResponse(status_code=400, content=exc.to_dict())

    app.include_router(generate_routes.router, prefix="/api")
    return app, job_manager


def test_parse_prompt_accepts_json_body(test_app):
    app, _ = test_app
    client = TestClient(app)

    response = client.post("/api/generate/parse-prompt", json={"prompt": "Make my character dance like in the video"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == GenerationMode.VACE_MOTION_TRANSFER.value
    assert body["confidence"] > 0


def test_generate_rejects_pose_transfer_without_video(test_app):
    app, _ = test_app
    client = TestClient(app)

    response = client.post(
        "/api/generate",
        json={
            "reference_image_id": "reference",
            "prompt": "Transfer the motion to my character",
            "mode": GenerationMode.VACE_POSE_TRANSFER.value,
            "quality": QualityPreset.STANDARD.value,
        },
    )

    assert response.status_code == 400
    assert "requires an input video" in response.json()["message"]


def test_generate_auto_applies_parsed_parameters(test_app):
    app, job_manager = test_app
    client = TestClient(app)

    response = client.post(
        "/api/generate",
        json={
            "reference_image_id": "reference",
            "prompt": "Generate a cinematic 4k video for 5 seconds at 30 fps",
            "mode": GenerationMode.AUTO.value,
            "quality": QualityPreset.STANDARD.value,
        },
    )

    assert response.status_code == 200
    body = response.json()
    request = body["request"]
    assert request["mode"] == GenerationMode.WAN_R2V.value
    assert request["duration"] == 5
    assert request["fps"] == 30
    assert request["resolution"] == [3840, 2160]
    assert request["quality"] == QualityPreset.ULTRA.value

    created_job = job_manager.get_job(body["id"])
    assert created_job.status == JobStatus.PENDING
    assert created_job.request.mode == GenerationMode.WAN_R2V
    assert created_job.request.fps == 30


def test_generate_modes_surface_requirements(test_app):
    app, _ = test_app
    client = TestClient(app)

    response = client.get("/api/generate/modes")

    assert response.status_code == 200
    modes = {mode["value"]: mode for mode in response.json()["modes"]}
    assert modes[GenerationMode.VACE_MOTION_TRANSFER.value]["requires_input_video"] is True
    assert modes[GenerationMode.WAN_R2V.value]["supports_input_video"] is False
