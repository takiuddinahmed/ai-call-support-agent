import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
)
from starlette.websockets import WebSocketDisconnect

from support_agent.config import Settings
from support_agent.server.app import create_app
from support_agent.voice.asterisk import (
    AsteriskFrameSerializer,
    MediaStart,
    caller_from,
    parse_media_start,
)
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

@pytest.fixture
def app():
    return create_app(Settings(_env_file=None), store=FakeStore(), llm=FakeLlm([]),
                      models=FakeModels())


def asterisk_client(app):
    return TestClient(app, base_url="http://host.docker.internal:8100",
                      client=("127.0.0.1", 50000))


def test_host_docker_internal_is_allowed_by_default():
    assert "host.docker.internal" in Settings(_env_file=None).allowed_hosts


def refused_code(client, headers=None) -> int:
    """Opens a connection that the route must refuse. Returns the close code."""
    with pytest.raises(WebSocketDisconnect) as refused, \
            client.websocket_connect(PATH, headers=headers or {}) as ws:
        ws.receive_text()
    return refused.value.code


def test_second_call_is_rejected(app):
    client = asterisk_client(app)
    # The first connection waits for MEDIA_START, so a call is in progress.
    with client.websocket_connect(PATH):
        assert refused_code(client) == 1013


def test_phone_call_is_rejected_during_a_browser_call(app):
    app.state.voice_hub.start_call("browser", "guest")
    assert refused_code(asterisk_client(app)) == 1013


def test_browser_offer_is_refused_during_a_phone_call(app):
    client = asterisk_client(app)
    with client.websocket_connect(PATH):
        local = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))
        sid = local.post("/api/session", json={"channel": "voice"}).json()["session_id"]
        response = local.post("/api/voice/offer",
                              json={"session_id": sid, "sdp": "x", "type": "offer"})
        assert response.status_code == 409


def test_connection_from_a_web_page_is_refused(app):
    assert refused_code(asterisk_client(app), {"origin": "http://evil.example"}) == 1008


def test_wrong_audio_format_hangs_up(app):
    with asterisk_client(app).websocket_connect(PATH) as ws:
        ws.send_text(PLAIN_START.replace("slin16", "ulaw"))
        assert ws.receive_text() == "HANGUP"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    assert app.state.voice_hub.calls() == []


def test_route_is_free_again_after_a_failed_start(app):
    client = asterisk_client(app)
    with client.websocket_connect(PATH) as ws:
        ws.send_text("HANGUP")
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    with client.websocket_connect(PATH) as ws:
        ws.send_text(PLAIN_START.replace("slin16", "ulaw"))
        assert ws.receive_text() == "HANGUP"


# Docker Desktop forwards a container connection to the server as client 127.0.0.1. So
# the Host host.docker.internal can open only the phone WebSocket.

@pytest.mark.parametrize("path", ["/", "/voice", "/api/voice/calls", "/api/voice/events",
                                  "/api/models"])
def test_docker_host_name_gets_only_the_phone_route(app, path):
    assert asterisk_client(app).get(path).status_code == 403


def test_docker_host_name_cannot_post_to_the_api(app):
    response = asterisk_client(app).post("/api/session", json={"channel": "chat"})
    assert response.status_code == 403
