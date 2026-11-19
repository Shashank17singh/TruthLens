"""Application configuration using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )

    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "TruthLens"

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Security
    SECRET_KEY: str = "truthlens-dev-jwt-secret-key-change-in-production-32chars"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    HMAC_KEY: str = "truthlens-dev-hmac-key-change-in-production-32chars"
    HMAC_KEY_ID: str = "k1"

    # Database
    DATABASE_URL: str = "sqlite:///./truthlens.db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Storage (MinIO)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "truthlens-media"
    MINIO_SECURE: bool = False
    STORAGE_LOCAL_FALLBACK_DIR: str = "./data/storage"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_ALWAYS_EAGER: bool = False

    # Input Limits
    MAX_IMAGE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_AUDIO_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MAX_AUDIO_DURATION_SECONDS: float = 300.0  # 5 minutes
    MAX_TEXT_CHARACTERS: int = 20_000
    MAX_REPORT_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # Rate Limits (requests per minute)
    RATE_LIMIT_USER: int = 60
    RATE_LIMIT_API_KEY_DEFAULT: int = 30
    RATE_LIMIT_UNAUTHENTICATED: int = 10

    # Paths
    MODELS_YAML_PATH: str = "backend/models.yaml"

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]


settings = Settings()
