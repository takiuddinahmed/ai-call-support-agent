import httpx

from support_agent.loop import run_turn
from support_agent.session import Identity, Session
from support_agent.tools import ToolRunner
from tests.unit.fakes import FakeLlm, FakeStore


async def collect(session, llm, text="hi", max_rounds=6):
    runner = ToolRunner(FakeStore(), session)
    return [e async for e in run_turn(session, text, llm, runner, "m", max_rounds)]


def types(events):
    return [e.type for e in events if e.type != "text"]


async def test_plain_answer_event_order():
    session = Session.new(Identity.guest())
    events = await collect(session, FakeLlm(["Hello there"]))
    assert types(events) == ["turn_start", "llm_start", "first_token", "llm_end", "answer"]
    assert events[-1].data["text"].strip() == "Hello there"
    assert session.messages[-1] == {"role": "assistant", "content": "Hello there"}
    assert all(b.t_ms >= a.t_ms for a, b in zip(events, events[1:]))


async def test_tool_round_then_answer():
    session = Session.new(Identity.guest())
    llm = FakeLlm([[("get_product", '{"handle": "esp32-devkit-v1"}')], "It costs 550 taka"])
    events = await collect(session, llm)
    assert types(events) == ["turn_start", "llm_start", "first_token", "llm_end", "tool_start",
                             "tool_end", "llm_start", "first_token", "llm_end", "answer"]
    tool_end = next(e for e in events if e.type == "tool_end")
    assert tool_end.data["ok"] is True and "550" in tool_end.data["content"]
    second_call = llm.seen[1]
    assert second_call[-1]["role"] == "tool" and second_call[-1]["tool_call_id"] == "call_0"
    assert second_call[0]["role"] == "system"


async def test_too_many_tool_rounds():
    session = Session.new(Identity.guest())
    llm = FakeLlm([[("get_product", '{"handle": "x"}')]] * 7)
    events = await collect(session, llm, max_rounds=6)
    assert events[-1].type == "error"
    assert "too many tool calls" in events[-1].data["message"]


async def test_model_server_down_gives_one_error():
    session = Session.new(Identity.guest())
    events = await collect(session, FakeLlm([], error=httpx.ConnectError("refused")))
    assert types(events) == ["turn_start", "llm_start", "error"]
    assert "model server" in events[-1].data["message"]


async def test_customer_turns_count():
    session = Session.new(Identity.guest())
    await collect(session, FakeLlm(["a"]))
    await collect(session, FakeLlm(["b"]))
    assert session.customer_turns == 2
    assert [m["role"] for m in session.messages] == ["user", "assistant", "user", "assistant"]
