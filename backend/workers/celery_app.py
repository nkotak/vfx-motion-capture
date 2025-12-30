"""
Celery application configuration.
"""

from celery import Celery
from backend.core.config import settings

# Create Celery app
celery_app = Celery(
    "vfx_motion_capture",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["backend.workers.video_tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 minutes soft limit

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for GPU tasks
    worker_concurrency=1,  # Single worker for GPU

    # Result backend settings
    result_expires=86400,  # 24 hours

    # Task routing
    task_routes={
        "backend.workers.video_tasks.*": {"queue": "video"},
    },

    # Beat schedule (for periodic tasks)
    beat_schedule={
        "cleanup-expired-files": {
            "task": "backend.workers.video_tasks.cleanup_expired",
            "schedule": 3600.0,  # Every hour
        },
    },
)


# Optional: Configure for specific GPU
celery_app.conf.task_annotations = {
    "backend.workers.video_tasks.process_video_generation": {
        "rate_limit": "3/m",  # Max 3 per minute
    },
}
