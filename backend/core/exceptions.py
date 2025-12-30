"""
Custom exceptions for VFX Motion Capture application.
"""

from typing import Optional, Dict, Any


class VFXException(Exception):
    """Base exception for VFX Motion Capture."""

    def __init__(
        self,
        message: str,
        code: str = "VFX_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class VideoProcessingError(VFXException):
    """Error during video processing (encoding, decoding, etc.)."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="VIDEO_PROCESSING_ERROR",
            details=details
        )


class InvalidVideoError(VFXException):
    """Input video is invalid or unsupported."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="INVALID_VIDEO",
            details=details
        )


class InvalidImageError(VFXException):
    """Input image is invalid or unsupported."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="INVALID_IMAGE",
            details=details
        )


class ModelNotLoadedError(VFXException):
    """Required AI model is not loaded or available."""

    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Model '{model_name}' is not loaded or available",
            code="MODEL_NOT_LOADED",
            details={"model_name": model_name, **(details or {})}
        )


class InvalidInputError(VFXException):
    """Invalid input parameters."""

    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        extra_details = {"field": field} if field else {}
        super().__init__(
            message=message,
            code="INVALID_INPUT",
            details={**extra_details, **(details or {})}
        )


class FileNotFoundError(VFXException):
    """Requested file not found."""

    def __init__(self, file_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"File with ID '{file_id}' not found",
            code="FILE_NOT_FOUND",
            details={"file_id": file_id, **(details or {})}
        )


class JobNotFoundError(VFXException):
    """Requested job not found."""

    def __init__(self, job_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Job with ID '{job_id}' not found",
            code="JOB_NOT_FOUND",
            details={"job_id": job_id, **(details or {})}
        )


class JobCancelledError(VFXException):
    """Job was cancelled."""

    def __init__(self, job_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Job '{job_id}' was cancelled",
            code="JOB_CANCELLED",
            details={"job_id": job_id, **(details or {})}
        )


class ResourceExhaustedError(VFXException):
    """System resources exhausted (GPU memory, disk space, etc.)."""

    def __init__(self, resource: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="RESOURCE_EXHAUSTED",
            details={"resource": resource, **(details or {})}
        )


class RateLimitError(VFXException):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class PoseExtractionError(VFXException):
    """Error during pose extraction."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="POSE_EXTRACTION_ERROR",
            details=details
        )


class FaceDetectionError(VFXException):
    """Error during face detection."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="FACE_DETECTION_ERROR",
            details=details
        )


class NoFaceDetectedError(VFXException):
    """No face detected in image/video."""

    def __init__(self, source: str = "image", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"No face detected in the {source}",
            code="NO_FACE_DETECTED",
            details={"source": source, **(details or {})}
        )


class NoPoseDetectedError(VFXException):
    """No pose/person detected in image/video."""

    def __init__(self, source: str = "image", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"No person/pose detected in the {source}",
            code="NO_POSE_DETECTED",
            details={"source": source, **(details or {})}
        )
