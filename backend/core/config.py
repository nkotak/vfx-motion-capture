"""
Configuration management for VFX Motion Capture backend.
Uses pydantic-settings for environment variable parsing.
"""

from pathlib import Path
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "VFX Motion Capture"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    upload_dir: Path = Field(default=None)
    output_dir: Path = Field(default=None)
    models_dir: Path = Field(default=None)
    temp_dir: Path = Field(default=None)

    @field_validator("upload_dir", "output_dir", "models_dir", "temp_dir", mode="before")
    @classmethod
    def set_default_paths(cls, v, info):
        if v is None:
            base = Path(__file__).parent.parent.parent
            field_name = info.field_name
            if field_name == "upload_dir":
                return base / "uploads"
            elif field_name == "output_dir":
                return base / "outputs"
            elif field_name == "models_dir":
                return base / "models"
            elif field_name == "temp_dir":
                return base / "temp"
        return Path(v) if isinstance(v, str) else v

    # Redis / Celery
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def celery_broker_url(self) -> str:
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        return self.redis_url

    # Database
    database_url: str = "sqlite+aiosqlite:///./vfx_motion_capture.db"

    # Video Processing
    max_video_duration: int = 60  # seconds
    max_video_size_mb: int = 500
    max_image_size_mb: int = 50
    default_fps: int = 24
    default_resolution: tuple = (1280, 720)
    supported_video_formats: List[str] = [".mp4", ".mov", ".mpeg", ".avi", ".webm", ".mkv"]
    supported_image_formats: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    # Model Settings
    default_generation_mode: str = "vace_pose_transfer"
    liveportrait_model_path: Optional[str] = None
    wan_vace_model_path: Optional[str] = None
    wan_r2v_model_path: Optional[str] = None

    # Real-time Settings
    realtime_fps: int = 30
    realtime_buffer_size: int = 5
    realtime_max_latency_ms: int = 100

    # Hardware
    device: str = "auto"  # auto, cuda, cpu, mps
    gpu_memory_fraction: float = 0.9
    enable_fp16: bool = True
    enable_xformers: bool = True

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    max_concurrent_jobs: int = 3

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [self.upload_dir, self.output_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
