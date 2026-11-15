"""HTTP routes of the browser voice call: the page, the WebRTC offer, and the debug events.

The server imports this module at start. So this module imports Pipecat and the voice
pipeline only inside the offer handler, at the first call.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from support_agent.session import Session
from support_agent.voice.debug_hub import DebugHub
from support_agent.voice.deps import VoiceDeps

STATIC = Path(__file__).parent.parent / "server" / "static"
# The browser call uses 16 kHz audio, the rate that Whisper reads.
BROWSER_AUDIO_RATE = 16000
# If the WebRTC connection is not "connected" in this time after the answer, the call ends.
CONNECT_TIMEOUT_S = 15.0
# The debug routes show transcripts and order details. Only this computer can read them.
LOCAL_CLIENTS = {"127.0.0.1", "::1"}

CancelCall = Callable[[str], Awaitable[None]]


class OfferIn(BaseModel):
    session_id: str
    sdp: str
    type: str


class SessionRef(BaseModel):
    session_id: str


def caller_name(session: Session) -> str:
    """The caller name for the debug hub: the first name of the customer, or "guest"."""
    customer = session.identity.customer
    if session.identity.kind == "customer" and customer is not None:
        return customer.first_name
    return "guest"


def require_local_client(request: Request) -> None:
    host = request.client.host if request.client else None
    if host not in LOCAL_CLIENTS:
        raise HTTPException(403, "The debug data is only for this computer.")


def watch_connection(connection, cancel: CancelCall,
                     connect_timeout_s: float = CONNECT_TIMEOUT_S) -> asyncio.Task:
    """Ends the call if the WebRTC connection closes, fails, or does not connect in time.

    In Pipecat 1.16.0, SmallWebRTCConnection calls its handlers with the name of each new
    peer connection state, also "closed" and "failed". A connection that stays in
    "connecting" closes itself only after 60 s.
    """

    async def on_end(_connection) -> None:
        await cancel("The WebRTC connection ended.")

    connection.add_event_handler("closed", on_end)
    connection.add_event_handler("failed", on_end)

    async def check_connected() -> None:
        await asyncio.sleep(connect_timeout_s)
        if connection.pc.connectionState != "connected":
            logger.warning("The WebRTC connection did not connect in time. The call ends.")
            await cancel("The WebRTC connection did not connect.")

    return asyncio.create_task(check_connected())


def voice_router(deps: VoiceDeps, sessions: dict[str, Session], hub: DebugHub) -> APIRouter:
    router = APIRouter()
    webrtc = None
    # True from the start of an offer to the end of its call. With the hub check, it
    # allows one call at a time.
    busy = False
    # The running tasks of the calls. The set keeps a reference until each task ends.
    tasks: set[asyncio.Task] = set()

    def call_is_active() -> bool:
        return busy or any(call["ended_at"] is None for call in hub.calls())

    def session_for(session_id: str) -> Session:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "Unknown session. Start a new conversation.")
        return session

    def keep(task: asyncio.Task) -> None:
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def run_call(worker, call_id: str, watcher: asyncio.Task) -> None:
        nonlocal busy
        from pipecat.workers.runner import WorkerRunner

        try:
            await WorkerRunner(handle_sigint=False).run(worker)
        except Exception:  # noqa: BLE001  a failed call must not stop the server
            logger.exception(f"The voice call {call_id} failed")
        finally:
            watcher.cancel()
            hub.end_call(call_id)
            busy = False

    @router.get("/voice")
    async def voice_page():
        return FileResponse(STATIC / "voice.html")

    @router.post("/api/voice/new-call")
    async def new_call(body: SessionRef):
        """Starts a fresh conversation for the next call and keeps the login."""
        session = session_for(body.session_id)
        if call_is_active():
            raise HTTPException(409, "A call is already active. End it first.")
        session.start_over(session.identity)
        session.channel = "voice"
        return {"ok": True}

    @router.post("/api/voice/offer")
    async def offer(body: OfferIn):
        nonlocal busy, webrtc
        session = session_for(body.session_id)
        if call_is_active():
            raise HTTPException(409, "A call is already active. End it first.")
        busy = True
        worker = None
        connection_seen = None
        try:
            session.channel = "voice"
            await deps.models.ensure_loaded()
            from pipecat.transports.base_transport import TransportParams
            from pipecat.transports.smallwebrtc.request_handler import (
                SmallWebRTCRequest,
                SmallWebRTCRequestHandler,
            )
            from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

            from support_agent.voice.pipeline import build_task

            if webrtc is None:
                webrtc = SmallWebRTCRequestHandler()

            async def on_connection(connection) -> None:
                nonlocal worker, connection_seen
                connection_seen = connection
                transport = SmallWebRTCTransport(
                    webrtc_connection=connection,
                    params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
                )
                call_id = hub.start_call("browser", caller_name(session))
                try:
                    task = build_task(transport, session, deps, hub, call_id, BROWSER_AUDIO_RATE)
                except Exception:
                    hub.end_call(call_id)
                    raise

                async def cancel(reason: str) -> None:
                    await task.cancel(reason=reason)

                @transport.event_handler("on_client_disconnected")
                async def on_client_disconnected(_transport, _client):
                    await cancel("The caller ended the call.")

                watcher = watch_connection(connection, cancel)
                keep(watcher)
                keep(asyncio.create_task(run_call(task, call_id, watcher)))
                worker = task

            # The handler logs an error of the callback and does not raise it.
            answer = await webrtc.handle_web_request(
                SmallWebRTCRequest(sdp=body.sdp, type=body.type), on_connection
            )
        except Exception:
            # A call that started must not run without a connection to the caller.
            if worker is not None:
                await worker.cancel(reason="The offer failed.")
            raise
        finally:
            if worker is None:
                busy = False
        if worker is None:
            if connection_seen is not None:
                await connection_seen.disconnect()
            raise HTTPException(500, "The call did not start. Read the server log.")
        return answer

    @router.get("/api/voice/events")
    async def events(request: Request):
        require_local_client(request)
        stream = hub.subscribe()

        async def sse():
            async with aclosing(stream) as hub_events:
                async for event in hub_events:
                    yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @router.get("/api/voice/calls")
    async def list_calls(request: Request):
        require_local_client(request)
        return hub.calls()

    return router
