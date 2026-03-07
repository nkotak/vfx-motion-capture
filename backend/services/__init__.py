"""Lazy exports for backend services."""

__all__ = [
    "JobManager",
    "FileManager",
    "VideoProcessor",
    "PoseExtractor",
    "FaceDetector",
    "PromptParser",
]


def __getattr__(name):
    if name == "VideoProcessor":
        from .video_processor import VideoProcessor
        return VideoProcessor
    if name == "PoseExtractor":
        from .pose_extractor import PoseExtractor
        return PoseExtractor
    if name == "FaceDetector":
        from .face_detector import FaceDetector
        return FaceDetector
    if name == "PromptParser":
        from .prompt_parser import PromptParser
        return PromptParser
    if name == "JobManager":
        from .job_manager import JobManager
        return JobManager
    if name == "FileManager":
        from .file_manager import FileManager
        return FileManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
