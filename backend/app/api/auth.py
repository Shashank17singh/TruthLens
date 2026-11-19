"""Authentication and account endpoints."""

from datetime import datetime
from typing import Annotated

from backend.app.core.auth import get_current_user, rate_limit_unauthenticated
from backend.app.core.config import settings
from backend.app.core.errors import APIError
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from backend.app.db.models import User
from backend.app.db.session import get_db
from fastapi import APIRouter, Cookie, Depends, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    role: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str | None = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str | None = None


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    email_clean = req.email.lower()
    existing = db.query(User).filter(User.email == email_clean).first()
    if existing:
        raise APIError(status_code=409, code="email_taken", message="An account with this email already exists.")

    user = User(
        email=email_clean,
        password_hash=hash_password(req.password),
        display_name=req.display_name,
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(rate_limit_unauthenticated)])
def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    email_clean = req.email.lower()
    user = db.query(User).filter(User.email == email_clean).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise APIError(status_code=401, code="invalid_credentials", message="Invalid email or password.")

    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    # Set secure HttpOnly cookie for refresh token
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    response: Response,
    refresh_token: Annotated[str | None, Cookie()] = None,
    req: RefreshTokenRequest | None = None,
    db: Session = Depends(get_db),
):
    token_to_use = refresh_token or (req.refresh_token if req else None)
    if not token_to_use:
        raise APIError(status_code=401, code="invalid_token", message="Missing refresh token.")

    try:
        payload = decode_token(token_to_use)
    except Exception as exc:
        raise APIError(status_code=401, code="invalid_token", message="Invalid or expired refresh token.") from exc

    if payload.get("type") != "refresh":
        raise APIError(status_code=401, code="invalid_token", message="Invalid token type.")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise APIError(status_code=401, code="invalid_token", message="User account not found.")

    new_access_token = create_access_token(user.id)
    new_refresh_token = create_refresh_token(user.id)

    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
    )

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response):
    response.delete_cookie(key="refresh_token")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return user
