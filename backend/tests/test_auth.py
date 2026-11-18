"""Tests for authentication, registration, login, and tokens."""

from fastapi.testclient import TestClient


def test_register_success(client: TestClient):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "newuser@test.org", "password": "SecurePassword123!", "display_name": "New User"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@test.org"
    assert data["display_name"] == "New User"
    assert data["role"] == "user"
    assert "id" in data


def test_register_duplicate_email(client: TestClient, test_user):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": test_user.email, "password": "AnyPassword123!"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "email_taken"


def test_login_success(client: TestClient, test_user):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "Password123!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "refresh_token" in response.cookies


def test_login_invalid_password(client: TestClient, test_user):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "WrongPassword!"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"


def test_get_me(client: TestClient, auth_headers):
    response = client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "researcher@truthlens.ai"
