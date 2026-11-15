import asyncio
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from support_agent.config import Settings
from support_agent.server.app import create_app
from support_agent.voice.routes import watch_connection
from tests.unit.fakes import FakeLlm, FakeModels, FakeStore

LOCAL = ("127.0.0.1", 50000)


@pytest.fixture
def llm():
    return FakeLlm(["Hello", "Hello again"])


@pytest.fixture
def app(llm):
    return create_app(Settings(_env_file=None), store=FakeStore(), llm=llm, models=FakeModels())


@pytest.fixture
def client(app):
    return TestClient(app, base_url="http://localhost", client=LOCAL)


def new_session(client):
    return client.post("/api/session", json={"channel": "chat"}).json()["session_id"]


def login(client, sid):
    return client.post("/api/login", json={"session_id": sid, "email": "rahim@example.com",
                                           "password": "Demo1234!"})


def test_voice_page_is_served(client):
    assert client.get("/voice").status_code == 200


def test_offer_needs_known_session(client):
    response = client.post("/api/voice/offer",
                           json={"session_id": "nope", "sdp": "x", "type": "offer"})
    assert response.status_code == 404


def test_calls_list_starts_empty(client):
    assert client.get("/api/voice/calls").json() == []


def test_second_call_is_refused_while_a_call_is_active(app, client):
    app.state.voice_hub.start_call("browser", "guest")
    sid = new_session(client)
    response = client.post("/api/voice/offer",
                           json={"session_id": sid, "sdp": "x", "type": "offer"})
    assert response.status_code == 409


def test_login_works_for_the_voice_page(client):
    assert login(client, new_session(client)).status_code == 200


# Security

def test_unknown_host_is_refused(app):
    client = TestClient(app, base_url="http://evil.example", client=LOCAL)
    assert client.get("/api/voice/calls").status_code == 400
    assert client.get("/").status_code == 400


def test_allowed_hosts_come_from_the_settings():
    settings = Settings(_env_file=None, allowed_hosts=["localhost", "agent.local"])
    app = create_app(settings, store=FakeStore(), llm=FakeLlm([]), models=FakeModels())
    client = TestClient(app, base_url="http://agent.local", client=LOCAL)
    assert client.get("/voice").status_code == 200


@pytest.mark.parametrize("path", ["/api/voice/calls", "/api/voice/events"])
def test_debug_routes_refuse_a_client_that_is_not_local(app, path):
    client = TestClient(app, base_url="http://localhost", client=("192.168.1.20", 50000))
    assert client.get(path).status_code == 403


def test_debug_routes_accept_the_ipv6_loopback(app):
    client = TestClient(app, base_url="http://localhost", client=("::1", 50000))
    assert client.get("/api/voice/calls").status_code == 200


def test_offer_puts_no_email_in_the_hub(app, client):
    from support_agent.voice.routes import caller_name

    sid = new_session(client)
    login(client, sid)
    name = caller_name(app.state.sessions[sid])
    assert name == "Rahim"
    assert "@" not in name


# Start time

def test_server_import_does_not_load_pipecat():
    code = ("import sys, support_agent.server.app; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('pipecat', 'faster_whisper', 'onnxruntime', 'kokoro_onnx', 'mlx_whisper')))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=True)
    assert result.stdout.strip() == "[]"


# New call

def test_new_call_clears_the_history_and_keeps_the_login(app, client, llm):
    sid = new_session(client)
    login(client, sid)
    client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    session = app.state.sessions[sid]
    assert session.messages
    assert client.post("/api/voice/new-call", json={"session_id": sid}).status_code == 200
    assert session.messages == []
    assert session.identity.kind == "customer"
    assert session.channel == "voice"


def test_new_call_needs_known_session(client):
    assert client.post("/api/voice/new-call", json={"session_id": "nope"}).status_code == 404


def test_new_call_is_refused_during_a_call(app, client):
    sid = new_session(client)
    app.state.voice_hub.start_call("browser", "guest")
    assert client.post("/api/voice/new-call", json={"session_id": sid}).status_code == 409


# Connection watch

class FakePc:
    def __init__(self, state):
        self.connectionState = state


class FakeConnection:
    def __init__(self, state="new"):
        self.pc = FakePc(state)
        self.handlers = {}

    def add_event_handler(self, name, handler):
        self.handlers.setdefault(name, []).append(handler)

    async def fire(self, name):
        for handler in self.handlers.get(name, []):
            await handler(self)


class CancelRecorder:
    def __init__(self):
        self.reasons = []

    async def __call__(self, reason):
        self.reasons.append(reason)


async def test_watch_cancels_a_call_that_does_not_connect():
    cancel = CancelRecorder()
    watch_connection(FakeConnection("connecting"), cancel, connect_timeout_s=0.05)
    await asyncio.sleep(0.15)
    assert len(cancel.reasons) == 1


async def test_watch_keeps_a_connected_call():
    cancel = CancelRecorder()
    watch_connection(FakeConnection("connected"), cancel, connect_timeout_s=0.05)
    await asyncio.sleep(0.15)
    assert cancel.reasons == []


@pytest.mark.parametrize("event", ["closed", "failed"])
async def test_watch_cancels_the_call_when_the_connection_ends(event):
    cancel = CancelRecorder()
    connection = FakeConnection("connected")
    watcher = watch_connection(connection, cancel, connect_timeout_s=10)
    await connection.fire(event)
    assert len(cancel.reasons) == 1
    watcher.cancel()


async def test_watch_ends_a_browser_call_at_the_maximum_length():
    cancel = CancelRecorder()
    watcher = watch_connection(FakeConnection("connected"), cancel, connect_timeout_s=0.02,
                               max_call_s=0.1)
    await asyncio.sleep(0.05)
    assert cancel.reasons == []
    await asyncio.sleep(0.15)
    assert cancel.reasons == ["The call reached the maximum length."]
    assert watcher.done()
