import json

from fastapi.testclient import TestClient

from support_agent.config import Settings
from support_agent.server.app import create_app
from tests.unit.fakes import FakeLlm, FakeModels, FakeStore


def make_client(rounds=("Hello",)):
    llm = FakeLlm(list(rounds))
    app = create_app(Settings(_env_file=None), store=FakeStore(), llm=llm, models=FakeModels())
    return TestClient(app), llm


def new_session(client, channel="chat"):
    return client.post("/api/session", json={"channel": channel}).json()["session_id"]


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_index_page():
    client, _ = make_client()
    assert client.get("/").status_code == 200


def test_login_success_and_failure():
    client, _ = make_client()
    sid = new_session(client)
    ok = client.post("/api/login", json={"session_id": sid, "email": "rahim@example.com",
                                         "password": "Demo1234!"})
    assert ok.status_code == 200 and ok.json()["first_name"] == "Rahim"
    bad = client.post("/api/login", json={"session_id": sid, "email": "rahim@example.com",
                                          "password": "x"})
    assert bad.status_code == 401
    assert bad.json()["message"] == "The email or the password is wrong."


def test_chat_streams_events():
    client, _ = make_client()
    sid = new_session(client)
    response = client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert response.headers["content-type"].startswith("text/event-stream")
    kinds = [e["type"] for e in events(response)]
    assert kinds[0] == "turn_start" and kinds[-1] == "answer"


def test_unknown_session_is_404():
    client, _ = make_client()
    assert client.post("/api/chat", json={"session_id": "nope", "message": "hi"}).status_code == 404
    assert client.post("/api/logout", json={"session_id": "nope"}).status_code == 404


def test_channel_in_chat_request_changes_session():
    client, llm = make_client()
    sid = new_session(client)
    client.post("/api/chat", json={"session_id": sid, "message": "hi", "channel": "voice"})
    assert "one to three short sentences" in llm.seen[0][0]["content"]


def test_models_list():
    client, _ = make_client()
    data = client.get("/api/models").json()
    assert data == {"models": ["qwen3.5:4b", "other:7b"], "default": "qwen3.5:4b-q4_K_M"}


def test_logout_returns_to_guest():
    client, llm = make_client(rounds=("a", "b"))
    sid = new_session(client)
    client.post("/api/login", json={"session_id": sid, "email": "rahim@example.com",
                                    "password": "Demo1234!"})
    client.post("/api/logout", json={"session_id": sid})
    client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert "not logged in" in llm.seen[0][0]["content"]
