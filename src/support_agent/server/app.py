"""Chat and voice server: login, chat with streamed agent events, the model list, and
browser voice calls."""

from contextlib import aclosing
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from support_agent.config import Settings, get_settings
from support_agent.llm import OpenAiCompatClient
from support_agent.loop import run_turn
from support_agent.session import Identity, Session
from support_agent.store.client import StoreAuthError, StoreClient
from support_agent.tools import ToolRunner
from support_agent.voice.debug_hub import DebugHub
from support_agent.voice.deps import VoiceDeps
from support_agent.voice.routes import voice_router

STATIC = Path(__file__).parent / "static"
PORT = 8100
# Asterisk in Docker connects with this host name. Docker Desktop forwards each container
# connection as client 127.0.0.1, so the local-client check of the debug routes cannot
# tell a container from this computer.
DOCKER_HOST_NAME = "host.docker.internal"


class SessionIn(BaseModel):
    channel: Literal["chat", "voice"] = "chat"


class LoginIn(BaseModel):
    session_id: str
    email: str
    password: str


class SessionRef(BaseModel):
    session_id: str


class ChatIn(BaseModel):
    session_id: str
    message: str
    model: str | None = None
    channel: Literal["chat", "voice"] | None = None


def create_app(settings: Settings, store=None, llm=None, models=None) -> FastAPI:
    store = store or StoreClient(settings)
    llm = llm or OpenAiCompatClient(settings)
    models = models or llm
    sessions: dict[str, Session] = {}
    app = FastAPI(title="Support agent")
    # A Host header check stops DNS rebinding: a hostile web page cannot reach this local
    # server under its own domain name.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    @app.middleware("http")
    async def docker_host_gets_only_the_phone_route(request: Request, call_next):
        # A convenience, not a security boundary: an HTTP request with the Docker host
        # name gets 403, so that the phone host name opens only the phone WebSocket. A
        # container can send any Host header and looks like 127.0.0.1. So the containers
        # on this computer are trusted local clients. The phone route has its own token.
        if request.url.hostname == DOCKER_HOST_NAME:
            return JSONResponse({"detail": "This host name is only for the phone route."},
                                403)
        return await call_next(request)

    hub = DebugHub()
    app.state.voice_hub = hub
    app.state.sessions = sessions
    # The voice routes share the sessions, so that login and logout work for the voice page.
    app.include_router(voice_router(VoiceDeps(settings, store, llm), sessions, hub))

    def session_for(session_id: str) -> Session:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "Unknown session. Start a new conversation.")
        return session

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/api/session")
    async def new_session(body: SessionIn):
        session = Session.new(Identity.guest(), body.channel)
        sessions[session.id] = session
        return {"session_id": session.id}

    @app.post("/api/login")
    async def login(body: LoginIn):
        session = session_for(body.session_id)
        try:
            token = await store.login(body.email.strip(), body.password)
        except StoreAuthError:
            return JSONResponse({"message": "The email or the password is wrong."}, 401)
        customer = await store.get_customer(token)
        session.start_over(Identity(kind="customer", token=token, customer=customer))
        return {"first_name": customer.first_name}

    @app.post("/api/logout")
    async def logout(body: SessionRef):
        session_for(body.session_id).start_over(Identity.guest())
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(body: ChatIn):
        session = session_for(body.session_id)
        if body.channel:
            session.channel = body.channel
        model = body.model or settings.llm_model

        async def stream():
            runner = ToolRunner(store, session)
            # aclosing ends the turn at once when the client closes the connection.
            async with aclosing(run_turn(
                session, body.message, llm, runner, model, settings.max_tool_rounds,
                reasoning_effort_after_tools=settings.llm_reasoning_effort_after_tools,
                max_prompt_tokens=settings.max_prompt_tokens,
            )) as events:
                async for event in events:
                    yield f"data: {event.to_json()}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/models")
    async def list_models():
        try:
            names = await models.list_models()
        except Exception:  # noqa: BLE001  the page still works with the default model
            names = []
        return {"models": names, "default": settings.llm_model}

    return app


def run() -> None:
    uvicorn.run(create_app(get_settings()), host="127.0.0.1", port=PORT)
