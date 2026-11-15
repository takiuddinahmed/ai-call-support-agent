import asyncio
import base64
import threading
import time

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from starlette.websockets import WebSocketDisconnect

from support_agent.config import Settings
from support_agent.server.app import create_app
from support_agent.voice.asterisk import (
    AsteriskFrameSerializer,
    MediaStart,
    caller_from,
    parse_media_start,
)
from support_agent.voice.deps import VoiceModels
from tests.unit.fakes import FakeLlm, FakeModels, FakeStore

# The TestClient uses the Host "testserver" for a WebSocket path, so the tests give
# the full URL with the host name that Asterisk uses.
PATH = "ws://host.docker.internal:8100/api/voice/asterisk"
PLAIN_START = ("MEDIA_START connection_id:agent channel:WebSocket/agent/0x1 channel_id:17.7 "
               "format:slin16 optimal_frame_size:640 ptime:20")


# Serializer

async def test_binary_audio_becomes_input_frame():
    frame = await AsteriskFrameSerializer().deserialize(b"\x00\x01" * 320)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.sample_rate == 16000 and frame.num_channels == 1
    assert frame.audio == b"\x00\x01" * 320


async def test_interruption_flushes_media():
    assert await AsteriskFrameSerializer().serialize(InterruptionFrame()) == "FLUSH_MEDIA"


async def test_hangup_ends_call():
    assert isinstance(await AsteriskFrameSerializer().deserialize("HANGUP"), EndFrame)


async def test_end_of_call_hangs_up():
    assert await AsteriskFrameSerializer().serialize(EndFrame()) == "HANGUP"


async def test_cancel_sends_nothing():
    assert await AsteriskFrameSerializer().serialize(CancelFrame()) is None


async def test_agent_audio_goes_out_as_raw_pcm():
    audio = b"\x01\x02" * 320
    frame = OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
    assert await AsteriskFrameSerializer().serialize(frame) == audio


@pytest.mark.parametrize("text", ["MEDIA_XOFF", "MEDIA_XON", "QUEUE_DRAINED", "STATUS x:1"])
async def test_other_control_messages_give_no_frame(text):
    assert await AsteriskFrameSerializer().deserialize(text) is None


async def test_media_start_gives_no_frame():
    assert await AsteriskFrameSerializer().deserialize(PLAIN_START) is None


# MEDIA_START

def test_plain_text_media_start_is_parsed():
    start = parse_media_start(PLAIN_START)
    assert start == MediaStart(format="slin16", optimal_frame_size=640, ptime=20,
                               channel="WebSocket/agent/0x1", caller=None)


def test_json_media_start_gives_the_caller_number():
    text = ('{"event": "MEDIA_START", "channel": "WebSocket/agent/0x1", "format": "slin16", '
            '"optimal_frame_size": 640, "ptime": 20, '
            '"channel_variables": {"CALLERID(num)": "1001"}}')
    start = parse_media_start(text)
    assert start.format == "slin16" and start.optimal_frame_size == 640
    assert start.caller == "1001"


@pytest.mark.parametrize("text", ["HANGUP", "", "{}", "MEDIA_XOFF"])
def test_other_messages_are_not_media_start(text):
    assert parse_media_start(text) is None


def test_caller_is_the_number_or_phone():
    assert caller_from(MediaStart("slin16", 640, 20, "c", "1001"), None) == "1001"
    assert caller_from(MediaStart("slin16", 640, 20, "c", None), "+8801711") == "+8801711"
    assert caller_from(MediaStart("slin16", 640, 20, "c", None), None) == "phone"


@pytest.mark.parametrize("value", ["rahim@example.com", "abc", "1" * 40, ""])
def test_caller_that_is_not_a_number_becomes_phone(value):
    assert caller_from(MediaStart("slin16", 640, 20, "c", value), None) == "phone"


# Route

TOKEN = "test-token-1234"


def basic(password: str, user: str = "asterisk") -> dict:
    """The Authorization header that Asterisk sends for username and password."""
    raw = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"authorization": f"Basic {raw}"}


def make_app(**settings):
    values = {"phone_token": TOKEN} | settings
    return create_app(Settings(_env_file=None, **values), store=FakeStore(), llm=FakeLlm([]),
                      models=FakeModels())


@pytest.fixture
def app():
    return make_app()


def asterisk_client(app):
    return TestClient(app, base_url="http://host.docker.internal:8100",
                      client=("127.0.0.1", 50000))


def open_call(client, path=PATH, headers=None, token=TOKEN):
    return client.websocket_connect(path, headers=basic(token) | (headers or {}))


def refused_code(client, headers=None, token=TOKEN) -> int:
    """Opens a connection that the route must refuse. Returns the close code."""
    with pytest.raises(WebSocketDisconnect) as refused, \
            open_call(client, headers=headers, token=token) as ws:
        ws.receive_text()
    return refused.value.code


def is_accepted(client, **kwargs) -> bool:
    """True if the route accepts the connection: a wrong format then gives HANGUP."""
    with open_call(client, **kwargs) as ws:
        ws.send_text(PLAIN_START.replace("slin16", "ulaw"))
        return ws.receive_text() == "HANGUP"


def test_host_docker_internal_is_allowed_by_default():
    assert "host.docker.internal" in Settings(_env_file=None).allowed_hosts


def test_phone_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.phone_token == ""
    assert settings.max_call_seconds == 900


def test_phone_token_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_PHONE_TOKEN", "from-env")
    assert Settings(_env_file=None).phone_token == "from-env"


# Shared secret

def test_right_token_is_accepted(app):
    assert is_accepted(asterisk_client(app))


def test_wrong_token_is_refused(app):
    assert refused_code(asterisk_client(app), token="wrong") == 1008


def test_missing_authorization_is_refused(app):
    client = asterisk_client(app)
    with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect(PATH) as ws:
        ws.receive_text()
    assert refused.value.code == 1008


@pytest.mark.parametrize("value", ["Bearer test-token-1234", "Basic !!!", "Basic"])
def test_bad_authorization_header_is_refused(app, value):
    client = asterisk_client(app)
    with pytest.raises(WebSocketDisconnect) as refused, \
            client.websocket_connect(PATH, headers={"authorization": value}) as ws:
        ws.receive_text()
    assert refused.value.code == 1008


def test_no_token_setting_refuses_every_call():
    app = make_app(phone_token="")
    client = asterisk_client(app)
    assert refused_code(client, token="") == 1008
    assert refused_code(client, token="anything") == 1008


def test_token_check_comes_before_the_busy_check(app):
    app.state.voice_hub.start_call("browser", "guest")
    assert refused_code(asterisk_client(app), token="wrong") == 1008


# One call at a time

def test_second_call_is_rejected(app):
    client = asterisk_client(app)
    # The first connection waits for MEDIA_START, so a call is in progress.
    with open_call(client):
        assert refused_code(client) == 1013


def test_phone_call_is_rejected_during_a_browser_call(app):
    app.state.voice_hub.start_call("browser", "guest")
    assert refused_code(asterisk_client(app)) == 1013


def test_browser_offer_is_refused_during_a_phone_call(app):
    client = asterisk_client(app)
    with open_call(client):
        local = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))
        sid = local.post("/api/session", json={"channel": "voice"}).json()["session_id"]
        response = local.post("/api/voice/offer",
                              json={"session_id": sid, "sdp": "x", "type": "offer"})
        assert response.status_code == 409


def test_connection_from_a_web_page_is_refused(app):
    assert refused_code(asterisk_client(app), {"origin": "http://evil.example"}) == 1008


# MEDIA_START

def test_wrong_audio_format_hangs_up(app):
    with open_call(asterisk_client(app)) as ws:
        ws.send_text(PLAIN_START.replace("slin16", "ulaw"))
        assert ws.receive_text() == "HANGUP"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    assert app.state.voice_hub.calls() == []


def test_route_is_free_again_after_a_failed_start(app):
    client = asterisk_client(app)
    with open_call(client) as ws:
        ws.send_text("HANGUP")
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    assert is_accepted(client)


def test_media_start_wait_has_one_deadline(app, monkeypatch):
    # The real values are a 5 s deadline and a binary message every 4 s.
    monkeypatch.setattr("support_agent.voice.routes.MEDIA_START_TIMEOUT_S", 0.5)
    with open_call(asterisk_client(app)) as ws:
        stop = threading.Event()

        def send_audio():
            while not stop.wait(0.4):
                ws.send_bytes(b"\x00" * 640)

        sender = threading.Thread(target=send_audio)
        started = time.monotonic()
        sender.start()
        try:
            assert ws.receive_text() == "HANGUP"
            elapsed = time.monotonic() - started
        finally:
            stop.set()
            sender.join()
    assert elapsed < 1.0


# Running calls, with a fake pipeline

class FakeTurn(FrameProcessor):
    """Starts a long agent turn at the first caller audio. The turn sends agent audio."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self._task = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            if self._task is None:
                self._task = self.create_task(self._speak())
            return
        await self.push_frame(frame, direction)

    async def _speak(self):
        self.started.set()
        try:
            while True:
                await self.push_frame(OutputAudioRawFrame(b"\x00" * 640, 16000, 1))
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def cleanup(self):
        if self._task is not None:
            await self.cancel_task(self._task)
        await super().cleanup()


class FakePipeline:
    def __init__(self):
        self.turn = None

    def build_task(self, transport, session, deps, hub, call_id, audio_rate):
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker

        self.turn = FakeTurn()
        return PipelineWorker(
            Pipeline([transport.input(), self.turn, transport.output()]),
            params=PipelineParams(audio_in_sample_rate=audio_rate,
                                  audio_out_sample_rate=audio_rate),
            enable_rtvi=False,
        )


@pytest.fixture
def fake_pipeline(monkeypatch):
    fake = FakePipeline()

    async def loaded(_self):
        return None

    monkeypatch.setattr(VoiceModels, "ensure_loaded", loaded)
    monkeypatch.setattr("support_agent.voice.pipeline.build_task", fake.build_task)
    return fake


def wait_for(check, seconds=3.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return check()


def test_hangup_during_a_turn_frees_the_call(app, fake_pipeline):
    client = asterisk_client(app)
    hub = app.state.voice_hub
    with open_call(client) as ws:
        ws.send_text(PLAIN_START)
        ws.send_bytes(b"\x00" * 640)
        assert len(ws.receive_bytes()) == 640
        assert fake_pipeline.turn.started.is_set()
        assert hub.calls()[-1]["ended_at"] is None
    # The caller hung up: Asterisk closed the WebSocket in the middle of the turn.
    assert wait_for(lambda: hub.calls()[-1]["ended_at"] is not None)
    assert fake_pipeline.turn.cancelled.is_set()
    assert hub.calls()[-1]["channel"] == "phone"
    assert is_accepted(client)


def test_caller_number_comes_from_the_uri(app, fake_pipeline):
    client = asterisk_client(app)
    with open_call(client, path=PATH + "?caller=%2B8801711") as ws:
        ws.send_text(PLAIN_START)
        ws.send_bytes(b"\x00" * 640)
        ws.receive_bytes()
        assert app.state.voice_hub.calls()[-1]["caller"] == "+8801711"


def test_caller_that_is_not_a_number_shows_phone(app, fake_pipeline):
    client = asterisk_client(app)
    with open_call(client, path=PATH + "?caller=rahim%40example.com") as ws:
        ws.send_text(PLAIN_START)
        ws.send_bytes(b"\x00" * 640)
        ws.receive_bytes()
        assert app.state.voice_hub.calls()[-1]["caller"] == "phone"


def test_call_ends_after_the_maximum_length(fake_pipeline):
    app = make_app(max_call_seconds=1)
    client = asterisk_client(app)
    started = time.monotonic()
    with open_call(client) as ws:
        ws.send_text(PLAIN_START)
        while True:
            message = ws.receive()
            if message.get("text") == "HANGUP":
                break
        elapsed = time.monotonic() - started
        with pytest.raises(WebSocketDisconnect):
            while True:
                ws.receive_text()
    assert 0.9 < elapsed < 3.0
    assert wait_for(lambda: app.state.voice_hub.calls()[-1]["ended_at"] is not None)
    assert is_accepted(client)


# Docker Desktop forwards a container connection to the server as client 127.0.0.1. So
# the Host host.docker.internal can open only the phone WebSocket.

@pytest.mark.parametrize("path", ["/", "/voice", "/api/voice/calls", "/api/voice/events",
                                  "/api/models"])
def test_docker_host_name_gets_only_the_phone_route(app, path):
    assert asterisk_client(app).get(path).status_code == 403


def test_docker_host_name_cannot_post_to_the_api(app):
    response = asterisk_client(app).post("/api/session", json={"channel": "chat"})
    assert response.status_code == 403
