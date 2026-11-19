"""WebSocket endpoint for real-time analysis progress and completion notifications."""

import asyncio
import json
import logging
import time

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from backend.app.api.analyses import ws_tickets
from backend.app.core.config import settings
from backend.app.db.models import Analysis
from backend.app.db.session import SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/analyses/{id}")
async def websocket_analysis_progress(websocket: WebSocket, id: str, ticket: str | None = None):
    # Validate ticket
    if not ticket or ticket not in ws_tickets:
        await websocket.close(code=4401, reason="Invalid or missing ticket")
        return

    expected_id, expires_at = ws_tickets.pop(ticket)
    if time.time() > expires_at or expected_id != id:
        await websocket.close(code=4401, reason="Ticket expired or mismatched")
        return

    # Check analysis exists
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == id).first()
        if not analysis:
            await websocket.close(code=4404, reason="Analysis not found")
            return
        initial_status = analysis.status
    finally:
        db.close()

    await websocket.accept()

    # If already finished, send terminal event immediately
    if initial_status == "completed":
        await websocket.send_json({"type": "completed", "analysis_id": id})
        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
        return
    elif initial_status == "failed":
        await websocket.send_json({"type": "failed", "analysis_id": id, "error": {"code": "failed", "message": "Analysis failed."}})
        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
        return

    # Connect to Redis pub/sub to stream events
    pubsub = None
    r = None
    try:
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = r.pubsub()
        channel = f"analysis:{id}"
        await pubsub.subscribe(channel)

        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("type") == "message":
                data = json.loads(message["data"])
                await websocket.send_json(data)
                if data.get("type") in ("completed", "failed"):
                    break
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client for analysis %s", id)
    except Exception as exc:
        logger.debug("Redis pub/sub not available or error (%s); polling DB fallback.", exc)
        # Polling fallback when Redis is not available
        try:
            for _ in range(60):
                db = SessionLocal()
                try:
                    analysis = db.query(Analysis).filter(Analysis.id == id).first()
                    if analysis:
                        if analysis.status == "completed":
                            await websocket.send_json({"type": "completed", "analysis_id": id})
                            break
                        elif analysis.status == "failed":
                            await websocket.send_json({
                                "type": "failed",
                                "analysis_id": id,
                                "error": {"code": analysis.error_code or "failed", "message": analysis.error_message or "Analysis failed."},
                            })
                            break
                        else:
                            await websocket.send_json({
                                "type": "progress",
                                "analysis_id": id,
                                "stage": "processing",
                                "progress": analysis.progress,
                            })
                finally:
                    db.close()
                await asyncio.sleep(1.0)
        except Exception:
            pass
    finally:
        if pubsub:
            try:
                await pubsub.unsubscribe()
                await pubsub.close()
            except Exception:
                pass
        if r:
            try:
                await r.close()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
