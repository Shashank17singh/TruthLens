"""Main FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from backend.app.api.admin import router as admin_router
from backend.app.api.analyses import router as analyses_router
from backend.app.api.analyze import router as analyze_router
from backend.app.api.api_keys import router as api_keys_router
from backend.app.api.auth import router as auth_router
from backend.app.api.health import router as health_router
from backend.app.api.models import router as models_router
from backend.app.api.reports import router as reports_router
from backend.app.api.websocket import router as websocket_router
from backend.app.core.config import settings
from backend.app.core.errors import APIError, api_error_handler, error_response
from backend.app.db.session import create_tables
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure database schema tables exist on startup
    create_tables()
    yield


app = FastAPI(
    title="TruthLens API",
    description="Multi-modal deepfake and AI-content detection platform API.",
    version="1.0.0",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
)

# Exception handlers
app.add_exception_handler(APIError, api_error_handler)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled server error: {exc}")
    return JSONResponse(
        status_code=500,
        content=error_response(
            code="internal_error",
            message="An unexpected server error occurred.",
        ),
    )


# Mount routers under /api/v1
app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(analyze_router, prefix=settings.API_V1_PREFIX)
app.include_router(analyses_router, prefix=settings.API_V1_PREFIX)
app.include_router(reports_router, prefix=settings.API_V1_PREFIX)
app.include_router(api_keys_router, prefix=settings.API_V1_PREFIX)
app.include_router(admin_router, prefix=settings.API_V1_PREFIX)
app.include_router(models_router, prefix=settings.API_V1_PREFIX)
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket_router, prefix=settings.API_V1_PREFIX)

# Also expose health probes at top-level
app.include_router(health_router)
