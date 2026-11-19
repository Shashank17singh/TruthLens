"""MinIO object storage client with automatic fallback for local development/testing."""

import io
import logging
from datetime import timedelta
from pathlib import Path

from minio import Minio

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    def __init__(self):
        self._minio_client: Minio | None = None
        self._bucket_name = settings.MINIO_BUCKET_NAME
        self._local_fallback_dir = Path(settings.STORAGE_LOCAL_FALLBACK_DIR)
        self._use_local = False
        self._init_client()

    def _init_client(self) -> None:
        try:
            import urllib3
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=0.2, read=0.5),
                retries=False,
            )
            client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
                http_client=http_client,
            )
            # Check connection
            if not client.bucket_exists(self._bucket_name):
                client.make_bucket(self._bucket_name)
            self._minio_client = client
            self._use_local = False
            logger.info("Connected to MinIO object storage successfully.")
        except Exception as exc:
            logger.warning(
                "Could not connect to MinIO at %s: %s. Using local filesystem storage at %s.",
                settings.MINIO_ENDPOINT,
                exc,
                self._local_fallback_dir,
            )
            self._use_local = True
            self._local_fallback_dir.mkdir(parents=True, exist_ok=True)

    def put_object(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Store bytes under key. Returns key."""
        if not self._use_local and self._minio_client:
            try:
                stream = io.BytesIO(data)
                self._minio_client.put_object(
                    self._bucket_name,
                    key,
                    stream,
                    length=len(data),
                    content_type=content_type,
                )
                return key
            except Exception as exc:
                logger.warning("MinIO put_object failed (%s). Falling back to local storage.", exc)
                self._use_local = True

        target_path = self._local_fallback_dir / key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        return key

    def get_object(self, key: str) -> bytes:
        """Retrieve bytes stored under key."""
        if not self._use_local and self._minio_client:
            try:
                response = self._minio_client.get_object(self._bucket_name, key)
                try:
                    return response.read()
                finally:
                    response.close()
                    response.release_conn()
            except Exception as exc:
                logger.warning("MinIO get_object failed (%s). Checking local storage.", exc)

        target_path = self._local_fallback_dir / key
        if not target_path.exists():
            raise FileNotFoundError(f"Storage object not found: {key}")
        return target_path.read_bytes()

    def delete_object(self, key: str) -> None:
        """Delete object stored under key."""
        if not self._use_local and self._minio_client:
            try:
                self._minio_client.remove_object(self._bucket_name, key)
            except Exception as exc:
                logger.warning("MinIO remove_object failed (%s).", exc)

        target_path = self._local_fallback_dir / key
        if target_path.exists():
            target_path.unlink()

    def get_presigned_url(self, key: str, expires_seconds: int = 300) -> str:
        """Get presigned download URL for key."""
        if not self._use_local and self._minio_client:
            try:
                return self._minio_client.presigned_get_object(
                    self._bucket_name,
                    key,
                    expires=timedelta(seconds=expires_seconds),
                )
            except Exception as exc:
                logger.warning("MinIO presigned_get_object failed (%s).", exc)

        return f"/api/v1/storage/local/{key}"


storage = StorageManager()
