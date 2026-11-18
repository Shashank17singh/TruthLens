"""Celery configuration with queues and hourly purge schedule."""

import logging

from backend.app.core.config import settings
from celery import Celery
from celery.signals import worker_process_init
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)

broker_url = "memory://" if settings.CELERY_TASK_ALWAYS_EAGER else settings.CELERY_BROKER_URL
backend_url = "cache+memory://" if settings.CELERY_TASK_ALWAYS_EAGER else settings.CELERY_RESULT_BACKEND

celery_app = Celery(
    "truthlens_worker",
    broker=broker_url,
    backend=backend_url,
)

# Queues: fast (image, text), audio, reports
default_exchange = Exchange("truthlens", type="direct")
celery_app.conf.task_queues = (
    Queue("fast", default_exchange, routing_key="fast"),
    Queue("audio", default_exchange, routing_key="audio"),
    Queue("reports", default_exchange, routing_key="reports"),
)

celery_app.conf.task_routes = {
    "backend.app.worker.tasks.analyze_image_task": {"queue": "fast"},
    "backend.app.worker.tasks.analyze_text_task": {"queue": "fast"},
    "backend.app.worker.tasks.analyze_url_task": {"queue": "fast"},
    "backend.app.worker.tasks.analyze_audio_task": {"queue": "audio"},
    "backend.app.worker.tasks.render_report_task": {"queue": "reports"},
    "backend.app.worker.purge.purge_expired_media_task": {"queue": "fast"},
}

celery_app.conf.update(
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=True,
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "purge-expired-media-hourly": {
            "task": "backend.app.worker.purge.purge_expired_media_task",
            "schedule": 3600.0,  # every hour
        },
    },
)


@worker_process_init.connect
def init_worker(**kwargs):
    """Load ONNX runtime sessions and model files once after worker forks."""
    logger.info("Initializing worker process: pre-loading models from registry.")
    try:
        from backend.app.inference.registry import registry
        registry.get_image_session()
        registry.get_audio_session()
        logger.info("Worker process initialized and models cached.")
    except Exception as exc:
        logger.warning("Could not pre-load all models on worker startup: %s", exc)
