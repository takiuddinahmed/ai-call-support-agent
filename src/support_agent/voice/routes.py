"""HTTP routes of the browser voice call: the page, the WebRTC offer, and the debug events."""

import asyncio
import json
from contextlib import aclosing
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner
from pydantic import BaseModel

from support_agent.session import Session
from support_agent.voice.debug_hub import DebugHub
from support_agent.voice.pipeline import VoiceDeps, build_task

STATIC = Path(__file__).parent.parent / "server" / "static"
# The browser call uses 16 kHz audio, the rate that Whisper reads.
BROWSER_AUDIO_RATE = 16000


class OfferIn(BaseModel):
    session_id: str
    sdp: str
    type: str


def _caller(session: Session) -> str:
    customer = session.identity.customer
    if session.identity.kind == "customer" and customer is not None:
        return customer.email
    return "guest"


def voice_router(deps: VoiceDeps, sessions: dict[str, Session], hub: DebugHub) -> APIRouter:
    router = APIRouter()
    webrtc = SmallWebRTCRequestHandler()
    # True from the start of an offer to the end of its call. With the hub check, it
    # allows one call at a time.
    busy = False
    # The running calls. The set keeps a reference to each task until the call ends.
    calls: set[asyncio.Task] = set()

    def call_is_active() -> bool:
        return busy or any(call["ended_at"] is None for call in hub.calls())

    async def run_call(task, call_id: str) -> None:
        nonlocal busy
        try:
            await WorkerRunner(handle_sigint=False).run(task)
        except Exception:  # noqa: BLE001  a failed call must not stop the server
            logger.exception(f"The voice call {call_id} failed")
        finally:
            hub.end_call(call_id)
            busy = False

    @router.get("/voice")
    async def voice_page():
        return FileResponse(STATIC / "voice.html")

    @router.post("/api/voice/offer")
    async def offer(body: OfferIn):
        nonlocal busy
        session = sessions.get(body.session_id)
        if session is None:
            raise HTTPException(404, "Unknown session. Start a new conversation.")
        if call_is_active():
            raise HTTPException(409, "A call is already active. End it first.")
        busy = True
        started = False
        connection_seen: SmallWebRTCConnection | None = None
        try:
            session.channel = "voice"
            await deps.models.ensure_loaded()

            async def on_connection(connection: SmallWebRTCConnection) -> None:
                nonlocal started, connection_seen
                connection_seen = connection
                transport = SmallWebRTCTransport(
                    webrtc_connection=connection,
                    params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
                )
                call_id = hub.start_call("browser", _caller(session))
                try:
                    task = build_task(transport, session, deps, hub, call_id, BROWSER_AUDIO_RATE)
                except Exception:
                    hub.end_call(call_id)
                    raise

                @transport.event_handler("on_client_disconnected")
                async def on_client_disconnected(_transport, _client):
                    await task.cancel(reason="The caller ended the call.")

                call = asyncio.create_task(run_call(task, call_id))
                calls.add(call)
                call.add_done_callback(calls.discard)
                started = True

            # The handler logs an error of the callback and does not raise it.
            answer = await webrtc.handle_web_request(
                SmallWebRTCRequest(sdp=body.sdp, type=body.type), on_connection
            )
        finally:
            if not started:
                busy = False
        if not started:
            if connection_seen is not None:
                await connection_seen.disconnect()
            raise HTTPException(500, "The call did not start. Read the server log.")
        return answer

    @router.get("/api/voice/events")
    async def events():
        stream = hub.subscribe()

        async def sse():
            async with aclosing(stream) as hub_events:
                async for event in hub_events:
                    yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @router.get("/api/voice/calls")
    async def list_calls():
        return hub.calls()

    return router
