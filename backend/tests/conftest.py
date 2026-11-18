"""Pytest configuration and fixtures for backend tests."""

import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Force testing configuration
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "True"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import backend.app.reports.generator
import backend.app.worker.purge
import backend.app.worker.tasks
from backend.app.core.config import settings
from backend.app.core.security import (
    create_access_token,
    generate_api_key,
    hash_password,
)
from backend.app.db.base import Base
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_db
from backend.app.main import app
from backend.app.worker.celery_app import celery_app

# Set eager mode and in-memory broker
settings.CELERY_TASK_ALWAYS_EAGER = True
celery_app.conf.update(
    task_always_eager=True,
    task_eager_propagates=True,
    broker_url="memory://",
    result_backend="cache+memory://",
)

# In-memory SQLite for fast isolated tests
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Bind worker tasks to test database session
backend.app.worker.tasks.SessionLocal = TestingSessionLocal
backend.app.reports.generator.SessionLocal = TestingSessionLocal
backend.app.worker.purge.SessionLocal = TestingSessionLocal


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session: Session) -> User:
    user = User(
        email="researcher@truthlens.ai",
        password_hash=hash_password("Password123!"),
        display_name="Test Researcher",
        role="user",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_user(db_session: Session) -> User:
    admin = User(
        email="admin@truthlens.ai",
        password_hash=hash_password("AdminSecret123!"),
        display_name="System Admin",
        role="admin",
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)
    return admin


@pytest.fixture
def auth_headers(test_user: User) -> dict[str, str]:
    token = create_access_token(test_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(admin_user: User) -> dict[str, str]:
    token = create_access_token(admin_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_api_key(db_session: Session, test_user: User) -> tuple[str, ApiKey]:
    raw_key, prefix, key_hash = generate_api_key()
    key_row = ApiKey(
        user_id=test_user.id,
        name="Chrome Extension Key",
        key_prefix=prefix,
        key_hash=key_hash,
        rate_limit_per_min=30,
    )
    db_session.add(key_row)
    db_session.commit()
    db_session.refresh(key_row)
    return raw_key, key_row


@pytest.fixture
def api_key_headers(test_api_key: tuple[str, ApiKey]) -> dict[str, str]:
    raw_key, _ = test_api_key
    return {"X-API-Key": raw_key}
