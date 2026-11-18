"""SQLAlchemy ORM models matching schema.md."""

from datetime import UTC, datetime

from backend.app.core.security import generate_uuidv7
from backend.app.db.base import Base
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON


def utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuidv7)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=True)
    role = Column(String(20), nullable=False, default="user")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=generate_uuidv7)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    key_prefix = Column(String(16), nullable=False)
    key_hash = Column(String(64), nullable=False, unique=True, index=True)
    rate_limit_per_min = Column(Integer, nullable=False, default=30)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    user = relationship("User", back_populates="api_keys")
    analyses = relationship("Analysis", back_populates="api_key")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, default=generate_uuidv7)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    api_key_id = Column(String(36), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True, index=True)
    modality = Column(String(20), nullable=True)  # 'image', 'audio', 'text'
    source_type = Column(String(20), nullable=False, default="upload")  # 'upload', 'url'
    source_url = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="queued")  # 'queued', 'processing', 'completed', 'failed'
    progress = Column(Integer, nullable=False, default=0)
    file_name = Column(String(255), nullable=True)
    mime_type = Column(String(100), nullable=True)
    size_bytes = Column(BigInteger, nullable=True)
    sha256 = Column(String(64), nullable=True)
    object_key = Column(String(512), nullable=True)
    retain_media = Column(Boolean, nullable=False, default=False)
    media_expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    media_deleted_at = Column(DateTime(timezone=True), nullable=True)
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="analyses")
    api_key = relationship("ApiKey", back_populates="analyses")
    results = relationship("AnalysisResult", back_populates="analysis", cascade="all, delete-orphan")
    report = relationship("Report", back_populates="analysis", uselist=False, cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="analysis", uselist=False, cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    probability = Column(Float, nullable=False)
    decision_threshold = Column(Float, nullable=False)
    verdict = Column(String(20), nullable=False)  # 'likely_real', 'uncertain', 'likely_fake'
    confidence_low = Column(Float, nullable=False)
    confidence_high = Column(Float, nullable=False)
    explanation = Column(Text, nullable=False)
    details = Column(JSON, nullable=False, default=dict)
    artefacts = Column(JSON, nullable=False, default=dict)
    latency_ms = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    analysis = relationship("Analysis", back_populates="results")

    __table_args__ = (
        UniqueConstraint("analysis_id", "model_name", "model_version", name="uq_analysis_result_model"),
    )


class Report(Base):
    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, default=generate_uuidv7)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, unique=True)
    status = Column(String(20), nullable=False, default="queued")  # 'queued', 'rendering', 'ready', 'failed', 'expired'
    object_key = Column(String(512), nullable=True)
    sha256 = Column(String(64), nullable=True, index=True)
    hmac = Column(String(128), nullable=True)
    hmac_key_id = Column(String(50), nullable=True)
    public_token = Column(String(64), nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    ready_at = Column(DateTime(timezone=True), nullable=True)

    analysis = relationship("Analysis", back_populates="report")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    is_correct = Column(Boolean, nullable=False)
    reported_label = Column(String(20), nullable=True)  # 'real', 'fake', 'unsure'
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    analysis = relationship("Analysis", back_populates="feedback")
    user = relationship("User", back_populates="feedbacks")

    __table_args__ = (
        UniqueConstraint("analysis_id", "user_id", name="uq_feedback_analysis_user"),
    )
