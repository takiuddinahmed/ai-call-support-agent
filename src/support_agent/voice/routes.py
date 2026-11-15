"""Routes of the voice calls: the browser page, the WebRTC offer, the Asterisk phone
WebSocket, and the debug events.

The server imports this module at start. So this module imports Pipecat and the voice
pipeline only inside the call handlers, at the first call.
"""

import asyncio
import base64
import binascii
import hmac
import json
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from support_agent.session import Identity, Session
from support_agent.voice.debug_hub import DebugHub
from support_agent.voice.deps import VoiceDeps

STATIC = Path(__file__).parent.parent / "server" / "static"
# The browser call uses 16 kHz audio, the rate that Whisper reads.
BROWSER_AUDIO_RATE = 16000
# If the WebRTC connection is not "connected" in this time after the answer, the call ends.
CONNECT_TIMEOUT_S = 15.0
# The debug routes show transcripts and order details. Only this computer can read them.
LOCAL_CLIENTS = {"127.0.0.1", "::1"}
# Asterisk sends MEDIA_START right after the connection opens.
MEDIA_START_TIMEOUT_S = 5.0
# WebSocket close codes (RFC 6455).
CLOSE_POLICY = 1008
CLOSE_UNSUPPORTED = 1003
CLOSE_TRY_AGAIN = 1013

# slin16: 16000 samples of 2 bytes in one second, so 320 bytes in 10 ms.
SLIN16_10MS_BYTES = 320

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


def phone_frame_size(optimal_frame_size: int) -> int:
    """The size of one agent audio frame to Asterisk: a whole number of 10 ms parts.

    If MEDIA_START gives no usable size, the size is 640 bytes (20 ms of slin16).
    """
    parts = optimal_frame_size // SLIN16_10MS_BYTES
    return parts * SLIN16_10MS_BYTES if parts > 0 else 2 * SLIN16_10MS_BYTES


def phone_token_matches(authorization: str | None, token: str) -> bool:
    """True if the HTTP Basic password in the Authorization header is the phone token.

    Asterisk sends the "username" and "password" of websocket_client.conf as HTTP Basic
    authentication. The user name is not checked. An empty token never matches.
    """
    if not token or not authorization:
        return False
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "basic" or not value:
        return False
    try:
        decoded = base64.b64decode(value.strip(), validate=True)
    except (binascii.Error, ValueError):
        return False
    _user, sep, password = decoded.partition(b":")
    if not sep:
        return False
    return hmac.compare_digest(password, token.encode())


def require_local_client(request: Request) -> None:
    host = request.client.host if request.client else None
    if host not in LOCAL_CLIENTS:
        raise HTTPException(403, "The debug data is only for this computer.")


def watch_connection(connection, cancel: CancelCall,
                     connect_timeout_s: float = CONNECT_TIMEOUT_S,
                     max_call_s: float | None = None) -> asyncio.Task:
    """Ends the call if the WebRTC connection closes, fails, or does not connect in time.

    If max_call_s is set, the call also ends max_call_s after the start of the watch.

    In Pipecat 1.16.0, SmallWebRTCConnection calls its handlers with the name of each new
    peer connection state, also "closed" and "failed". A connection that stays in
    "connecting" closes itself only after 60 s.
    """

    async def on_end(_connection) -> None:
        await cancel("The WebRTC connection ended.")

    connection.add_event_handler("closed", on_end)
    connection.add_event_handler("failed", on_end)

    async def check_connected() -> None:
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.sleep(connect_timeout_s)
        if connection.pc.connectionState != "connected":
            logger.warning("The WebRTC connection did not connect in time. The call ends.")
            await cancel("The WebRTC connection did not connect.")
            return
        if max_call_s is None:
            return
        await asyncio.sleep(max(0.0, start + max_call_s - loop.time()))
        logger.info(f"The browser call reached {max_call_s} s. The call ends.")
        await cancel("The call reached the maximum length.")

    return asyncio.create_task(check_connected())


def voice_router(deps: VoiceDeps, sessions: dict[str, Session], hub: DebugHub) -> APIRouter:
    router = APIRouter()
    webrtc = None
    # True from the start of an offer to the end of its call. With the hub check, it
    # allows one call at a time.
    busy = False
    # True after the first warning about an empty phone token.
    warned_no_token = False
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
            runner = WorkerRunner(handle_sigint=False)
            await runner.add_workers(worker)
            await runner.run()
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

                watcher = watch_connection(connection, cancel,
                                           max_call_s=deps.settings.max_call_seconds)
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

    async def read_media_start(websocket: WebSocket):
        """Waits for MEDIA_START. Returns None and closes the connection if it is not valid.

        The wait has one deadline. Other messages before MEDIA_START do not extend it.
        """
        from support_agent.voice.asterisk import SLIN16, parse_control, parse_media_start

        loop = asyncio.get_running_loop()
        deadline = loop.time() + MEDIA_START_TIMEOUT_S
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(),
                                                 max(0.0, deadline - loop.time()))
            except TimeoutError:
                logger.warning("Asterisk sent no MEDIA_START. The phone call ends.")
                await websocket.send_text("HANGUP")
                await websocket.close()
                return None
            if message["type"] == "websocket.disconnect":
                return None
            text = message.get("text")
            if text is None:
                continue
            if parse_control(text)[0] == "HANGUP":
                await websocket.close()
                return None
            start = parse_media_start(text)
            if start is None:
                continue
            if start.format != SLIN16:
                logger.warning(f"The phone call uses the format {start.format!r}, not "
                               f"{SLIN16}. Use the dial option c(slin16). The call ends.")
                await websocket.send_text("HANGUP")
                await websocket.close(code=CLOSE_UNSUPPORTED)
                return None
            return start

    def phone_caller_allowed(websocket: WebSocket) -> bool:
        nonlocal warned_no_token
        token = deps.settings.phone_token
        if not token:
            if not warned_no_token:
                warned_no_token = True
                logger.warning("AGENT_PHONE_TOKEN is not set, so the phone route refuses "
                               "every call. Set the same token in the agent .env and the "
                               "demo-store .env.")
            return False
        if not phone_token_matches(websocket.headers.get("authorization"), token):
            logger.warning("A phone connection had no valid token. It is refused.")
            return False
        return True

    @router.websocket(deps.settings.voice_asterisk_path)
    async def asterisk_call(websocket: WebSocket):
        """One phone call from Asterisk chan_websocket. The caller is a guest."""
        nonlocal busy
        # Asterisk sends no Origin header. A web browser always sends one. So a web page
        # cannot start a phone call on this local server.
        if websocket.headers.get("origin"):
            await websocket.close(code=CLOSE_POLICY)
            return
        # The shared secret comes before the busy check, so that a stranger learns nothing.
        if not phone_caller_allowed(websocket):
            await websocket.close(code=CLOSE_POLICY)
            return
        # Before the accept, uvicorn answers HTTP 403. Then the Asterisk Dial fails, and
        # the dialplan plays the busy tone.
        if call_is_active():
            await websocket.close(code=CLOSE_TRY_AGAIN)
            return
        busy = True
        call_id = None
        try:
            await websocket.accept()
            start = await read_media_start(websocket)
            if start is None:
                return
            await deps.models.ensure_loaded()
            from pipecat.transports.websocket.fastapi import (
                FastAPIWebsocketParams,
                FastAPIWebsocketTransport,
            )
            from pipecat.workers.runner import WorkerRunner

            from support_agent.voice.asterisk import (
                SLIN16_RATE,
                AsteriskFrameSerializer,
                caller_from,
            )
            from support_agent.voice.pipeline import build_task

            frame_size = phone_frame_size(start.optimal_frame_size)
            transport = FastAPIWebsocketTransport(
                websocket,
                FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    serializer=AsteriskFrameSerializer(),
                    # The output transport sends one frame of frame_size bytes and then
                    # waits for its play time. So the Asterisk queue stays short, and
                    # FLUSH_MEDIA stops the agent audio at once.
                    audio_out_10ms_chunks=frame_size // SLIN16_10MS_BYTES,
                    fixed_audio_packet_size=frame_size,
                    # The Origin check is above.
                    allowed_origins=[],
                ),
            )
            session = Session.new(Identity.guest(), "voice")
            call_id = hub.start_call("phone",
                                     caller_from(start, websocket.query_params.get("caller")))
            logger.info(f"Phone call {call_id} started on {start.channel}")
            task = build_task(transport, session, deps, hub, call_id, SLIN16_RATE)

            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(_transport, _websocket):
                await task.cancel(reason="Asterisk closed the phone call.")

            async def end_at_limit() -> None:
                await asyncio.sleep(deps.settings.max_call_seconds)
                logger.info(f"Phone call {call_id} reached {deps.settings.max_call_seconds} s. "
                            "The agent hangs up.")
                try:
                    await websocket.send_text("HANGUP")
                except Exception:  # noqa: BLE001  the caller can hang up at the same time
                    logger.debug("The HANGUP command did not go out.")
                await task.cancel(reason="The call reached the maximum length.")

            limit = asyncio.create_task(end_at_limit())
            try:
                runner = WorkerRunner(handle_sigint=False)
                await runner.add_workers(task)
                await runner.run()
            finally:
                limit.cancel()
        except WebSocketDisconnect:
            logger.info("Asterisk closed the connection before the call started.")
        except Exception:  # noqa: BLE001  a failed call must not stop the server
            logger.exception("The phone call failed")
        finally:
            if call_id is not None:
                hub.end_call(call_id)
                logger.info(f"Phone call {call_id} ended")
            busy = False

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
