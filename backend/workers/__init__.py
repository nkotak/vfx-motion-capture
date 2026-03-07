"""Lazy exports for worker modules."""

__all__ = ["celery_app", "process_video_generation"]


def __getattr__(name):
    if name == "celery_app":
        from .celery_app import celery_app

        return celery_app
    if name == "process_video_generation":
        from .video_tasks import process_video_generation

        return process_video_generation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
