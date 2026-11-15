import pytest
from fastapi.testclient import TestClient

from support_agent.config import Settings
from support_agent.server.app import create_app
from tests.unit.fakes import FakeLlm, FakeModels, FakeStore


@pytest.fixture
def app():
    return create_app(Settings(_env_file=None), store=FakeStore(), llm=FakeLlm(["Hello"]),
                      models=FakeModels())


@pytest.fixture
def client(app):
    return TestClient(app)


def new_session(client):
    return client.post("/api/session", json={"channel": "chat"}).json()["session_id"]


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
    sid = new_session(client)
    response = client.post("/api/login", json={"session_id": sid, "email": "rahim@example.com",
                                               "password": "Demo1234!"})
    assert response.status_code == 200
