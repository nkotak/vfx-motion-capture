"""
Pydantic models for request/response schemas and data structures.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
import uuid


class JobStatus(str, Enum):
    """Status of a generation job."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    EXTRACTING_POSE = "extracting_pose"
    GENERATING = "generating"
    ENCODING = "encoding"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationMode(str, Enum):
    """Available generation modes."""
    VACE_POSE_TRANSFER = "vace_pose_transfer"
    VACE_MOTION_TRANSFER = "vace_motion_transfer"
    WAN_R2V = "wan_r2v"
    LIVEPORTRAIT = "liveportrait"
    DEEP_LIVE_CAM = "deep_live_cam"
    AUTO = "auto"  # Let the system decide based on prompt


class QualityPreset(str, Enum):
    """Quality presets for generation."""
    DRAFT = "draft"  # Fast, lower quality
    STANDARD = "standard"  # Balanced
    HIGH = "high"  # Slow, high quality
    ULTRA = "ultra"  # Very slow, maximum quality


class OutputFormat(str, Enum):
    """Output video format."""
    MP4 = "mp4"
    WEBM = "webm"
    GIF = "gif"


# ============ Request Models ============

class GenerateRequest(BaseModel):
    """Request to generate a video with person replacement."""

    reference_image_id: str = Field(
        ...,
        description="ID of the uploaded reference image"
    )
    input_video_id: Optional[str] = Field(
        None,
        description="ID of the uploaded input video (for video-to-video mode)"
    )
    prompt: str = Field(
        default="Replace the person in the video with the person from the reference image",
        description="Natural language instruction for the generation"
    )
    mode: GenerationMode = Field(
        default=GenerationMode.AUTO,
        description="Generation mode to use"
    )
    quality: QualityPreset = Field(
        default=QualityPreset.STANDARD,
        description="Quality preset"
    )
    duration: Optional[float] = Field(
        None,
        ge=1.0,
        le=60.0,
        description="Output video duration in seconds (None = match input)"
    )
    fps: int = Field(
        default=24,
        ge=12,
        le=60,
        description="Output frames per second"
    )
    resolution: Optional[tuple[int, int]] = Field(
        None,
        description="Output resolution (width, height)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP4,
        description="Output video format"
    )
    strength: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Transformation strength (0 = no change, 1 = full replacement)"
    )
    preserve_background: bool = Field(
        default=True,
        description="Whether to preserve the background from input video"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model-specific parameters"
    )

    @field_validator("resolution", mode="before")
    @classmethod
    def parse_resolution(cls, v):
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return tuple(v)
        if isinstance(v, str):
            parts = v.lower().replace("x", ",").split(",")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        raise ValueError("Resolution must be (width, height) tuple")


class RealtimeConfig(BaseModel):
    """Configuration for real-time camera mode."""

    reference_image_id: str = Field(
        ...,
        description="ID of the uploaded reference image/character"
    )
    mode: GenerationMode = Field(
        default=GenerationMode.LIVEPORTRAIT,
        description="Real-time processing mode"
    )
    target_fps: int = Field(
        default=30,
        ge=15,
        le=60,
        description="Target output FPS"
    )
    face_only: bool = Field(
        default=False,
        description="Only transform face (faster) vs full body"
    )
    smoothing: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Temporal smoothing factor"
    )
    enhance_face: bool = Field(
        default=True,
        description="Apply face enhancement"
    )


class UploadResponse(BaseModel):
    """Response after file upload."""

    id: str = Field(..., description="Unique file ID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (image/video)")
    size_bytes: int = Field(..., description="File size in bytes")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    resolution: Optional[tuple[int, int]] = Field(None, description="Width x Height")
    fps: Optional[float] = Field(None, description="Video FPS")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")


# ============ Job Models ============

class JobCreate(BaseModel):
    """Internal model for creating a job."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: GenerateRequest
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JobProgress(BaseModel):
    """Progress update for a job."""

    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=100.0, description="Progress percentage")
    current_step: str = Field(default="", description="Current processing step")
    eta_seconds: Optional[float] = Field(None, description="Estimated time remaining")
    preview_url: Optional[str] = Field(None, description="URL to preview frame")
    error: Optional[str] = Field(None, description="Error message if failed")


class JobResponse(BaseModel):
    """Full job information response."""

    id: str
    status: JobStatus
    progress: float = 0.0
    current_step: str = ""
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    request: GenerateRequest
    result_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============ Internal Models ============

class VideoMetadata(BaseModel):
    """Metadata extracted from a video file."""

    path: Path
    duration: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    file_size_bytes: int


class PoseData(BaseModel):
    """Pose keypoints for a single frame."""

    frame_index: int
    keypoints: List[Dict[str, Any]]  # List of detected poses
    confidence: float
    bbox: Optional[List[float]] = None  # Bounding box [x, y, w, h]


class FaceData(BaseModel):
    """Face detection data for a single frame."""

    frame_index: int
    bbox: List[float]  # [x1, y1, x2, y2]
    landmarks: List[List[float]]  # 5 or 68 point landmarks
    embedding: Optional[List[float]] = None
    confidence: float


class ProcessingResult(BaseModel):
    """Result from a processing operation."""

    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============ WebSocket Models ============

class WSMessage(BaseModel):
    """WebSocket message structure."""

    type: str  # "progress", "preview", "complete", "error"
    payload: Dict[str, Any]


class WSProgressPayload(BaseModel):
    """Payload for progress updates via WebSocket."""

    job_id: str
    status: JobStatus
    progress: float
    step: str
    eta_seconds: Optional[float] = None


class WSPreviewPayload(BaseModel):
    """Payload for preview frames via WebSocket."""

    job_id: str
    frame_index: int
    image_base64: str  # Base64 encoded preview image
