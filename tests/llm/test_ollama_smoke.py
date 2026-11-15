import pytest

from support_agent.config import Settings
from support_agent.llm import OpenAiCompatClient
from support_agent.loop import run_turn
from support_agent.session import Identity, Session
from support_agent.store.client import StoreClient
from support_agent.tools import ToolRunner

pytestmark = [pytest.mark.llm, pytest.mark.store]


async def test_esp32_question_uses_search_and_names_esp32():
    settings = Settings(_env_file=None)
    store = StoreClient(settings)
    session = Session.new(Identity.guest())
    events = [
        e
        async for e in run_turn(
            session, "Do you have an ESP32 board?", OpenAiCompatClient(settings),
            ToolRunner(store, session), "qwen3.5:4b", 6,
        )
    ]
    await store.aclose()
    for e in events:
        if e.type in ("first_token", "llm_end", "tool_start", "answer", "error"):
            print(e.to_json()[:300])
    tools = [e.data["name"] for e in events if e.type == "tool_start"]
    assert any(name in ("search_products", "get_product") for name in tools)
    assert events[-1].type == "answer"
    assert "ESP32" in events[-1].data["text"]
