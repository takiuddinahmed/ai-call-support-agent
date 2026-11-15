import asyncio
import json

import httpx
import pytest

from support_agent.loop import run_turn
from support_agent.session import Identity, Session
from support_agent.tools import ToolRunner
from tests.unit.fakes import RAHIM, FakeLlm, FakeStore


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
    assert types(events) == ["turn_start", "llm_start", "first_token", "tool_call_start",
                             "llm_end", "tool_start", "tool_end", "llm_start", "first_token",
                             "llm_end", "answer"]
    tool_end = next(e for e in events if e.type == "tool_end")
    assert tool_end.data["ok"] is True and "550" in tool_end.data["content"]
    second_call = llm.seen[1]
    assert second_call[-1]["role"] == "tool" and second_call[-1]["tool_call_id"] == "call_0"
    assert second_call[0]["role"] == "system"


async def test_round_limit_forces_a_final_answer_without_tools():
    session = Session.new(Identity.guest())
    llm = FakeLlm([[("get_product", '{"handle": "x"}')]] * 6 + ["Here is what I found"])
    events = await collect(session, llm, max_rounds=6)
    assert events[-1].type == "answer"
    assert events[-1].data["text"].strip() == "Here is what I found"
    assert len([e for e in events if e.type == "tool_start"]) == 6
    assert llm.tools_seen[-1] == []
    assert "Answer now" in llm.seen[-1][-1]["content"]


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


async def test_tool_call_start_comes_before_llm_end():
    session = Session.new(Identity.guest())
    llm = FakeLlm([[("search_products", '{"text":"esp32"}')], "It costs 550 taka."])
    events = await collect(session, llm, text="price of esp32")
    kinds = [e.type for e in events]
    assert kinds.count("tool_call_start") == 1
    assert kinds.index("tool_call_start") < kinds.index("llm_end")
    assert events[kinds.index("tool_call_start")].data == {"name": "search_products", "round": 1}


async def run_and_cancel(session, llm, runner, text, cancel_at):
    """Runs one turn in a task and cancels the task after the first event of type cancel_at."""
    events = []
    reached = asyncio.Event()

    async def consume():
        async for e in run_turn(session, text, llm, runner, "m", 6):
            events.append(e)
            if e.type == cancel_at:
                reached.set()

    task = asyncio.create_task(consume())
    await reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return events


def assert_history_valid(messages):
    """Every assistant tool call has exactly one tool message with its id."""
    call_ids = [c["id"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    tool_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert sorted(call_ids) == sorted(tool_ids)


async def customer_with_summary(store):
    session = Session.new(Identity(kind="customer", token="tok", customer=RAHIM), "chat")
    result = await ToolRunner(store, session).run(
        "prepare_order", json.dumps({"items": [{"handle": "esp32-devkit-v1", "quantity": 2}]})
    )
    return session, json.loads(result.content)["summary_id"]


async def test_cancel_during_stream_drops_partial_answer():
    session = Session.new(Identity.guest())
    llm = FakeLlm(["one two three four"], delay_s=0.05)
    runner = ToolRunner(FakeStore(), session)
    events = await run_and_cancel(session, llm, runner, "hi", "text")
    assert "answer" not in [e.type for e in events]
    assert session.messages == [{"role": "user", "content": "hi"}]


async def test_cancel_during_tool_keeps_result():
    store = FakeStore()
    store.place_delay_s = 0.1
    session, summary_id = await customer_with_summary(store)
    llm = FakeLlm([[("place_order", json.dumps({"summary_id": summary_id}))], "Done."])
    events = await run_and_cancel(session, llm, ToolRunner(store, session), "yes", "tool_start")
    assert "tool_end" not in [e.type for e in events]
    assert len(store.placed) == 1
    last = session.messages[-1]
    assert last["role"] == "tool" and json.loads(last["content"])["order"]["display_id"] == 10
    assert_history_valid(session.messages)


async def test_cancel_between_tools_fills_missing_results():
    store = FakeStore()
    store.place_delay_s = 0.1
    session, summary_id = await customer_with_summary(store)
    llm = FakeLlm([[("place_order", json.dumps({"summary_id": summary_id})),
                    ("track_shipment", '{"tracking_number": "DEMO-1234567890"}')], "Done."])
    await run_and_cancel(session, llm, ToolRunner(store, session), "yes", "tool_start")
    assert len(store.placed) == 1
    assert ("track", "DEMO-1234567890") not in store.calls
    assert session.messages[-1] == {"role": "tool", "tool_call_id": "call_1",
                                    "content": "Not run: the customer interrupted."}
    assert_history_valid(session.messages)

