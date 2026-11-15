"""Chat server: login, chat with streamed agent events, and the model list."""

from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from support_agent.config import Settings, get_settings
from support_agent.llm import OpenAiCompatClient
from support_agent.loop import run_turn
from support_agent.session import Identity, Session
from support_agent.store.client import StoreAuthError, StoreClient
from support_agent.tools import ToolRunner

STATIC = Path(__file__).parent / "static"
PORT = 8100


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
            async for event in run_turn(
                session, body.message, llm, runner, model, settings.max_tool_rounds
            ):
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
