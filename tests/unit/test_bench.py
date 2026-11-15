from support_agent.bench.runner import Case, check_turn, load_cases, normalize_answer, run_case
from support_agent.loop import AgentEvent
from tests.unit.fakes import FakeLlm, FakeStore


def ev(kind, **data):
    return AgentEvent(kind, 0.0, data)


def test_load_cases_reads_all_yaml(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "- id: c1\n  language: en\n  login: none\n  channel: chat\n"
        "  turns:\n    - user: hi\n      expect: {answer_contains_any: [hello]}\n"
    )
    cases = load_cases(str(tmp_path / "*.yaml"))
    assert [c.id for c in cases] == ["c1"]
    assert cases[0].turns[0]["expect"]["answer_contains_any"] == ["hello"]


def test_normalize_answer_maps_bangla_digits_and_commas():
    assert normalize_answer("দাম ১,১৫০ টাকা") == "দাম 1150 টাকা"
    assert normalize_answer("BDT 1,150") == "bdt 1150"


def test_check_turn_tools_and_text():
    events = [ev("tool_start", name="search_products"), ev("tool_end", name="search_products", ok=True),
              ev("answer", text="ESP32 costs 550 BDT")]
    assert check_turn({"tools_called": ["search_products"], "answer_contains_any": ["550"]}, events, {}) == []
    failures = check_turn({"tools_not_called": ["search_products"], "answer_not_contains": ["550"]},
                          events, {})
    assert len(failures) == 2


def test_check_turn_live_price_and_order_placed():
    events = [ev("tool_end", name="place_order", ok=True), ev("answer", text="Done, ৫৫০ taka")]
    expect = {"answer_contains_any": ["{live:price:esp32-devkit-v1}"], "order_placed": True}
    assert check_turn(expect, events, {"esp32-devkit-v1": 550}) == []
    assert check_turn({"order_placed": False}, events, {}) != []


def test_check_turn_error_event_fails():
    assert check_turn({}, [ev("error", message="down")], {}) == ["error: down"]


async def test_run_case_with_fake_model():
    case = Case(id="x", language="en", login="none", channel="chat",
                turns=[{"user": "esp32?", "expect": {"tools_called": ["get_product"],
                                                     "answer_contains_any": ["550"]}}])
    llm = FakeLlm([[("get_product", '{"handle": "esp32-devkit-v1"}')], "It is 550 BDT"])
    result = await run_case(case, "m", FakeStore(), llm, max_tool_rounds=6, run_id="r1")
    assert result.passed, result.failures
    assert result.turns[0]["first_token_ms"] is not None


def test_parse_footprint_reads_current_and_peak():
    from support_agent.bench.runner import parse_footprint

    text = "ollama [1]: 64-bit\n    phys_footprint: 12836802832 B\n    phys_footprint_peak: 14843433256 B\n"
    assert parse_footprint(text) == (12836802832, 14843433256)
    assert parse_footprint("nothing") == (0, 0)


async def test_hung_turn_fails_with_timeout():
    import asyncio

    class HangingLlm:
        async def stream(self, model, messages, tools):
            await asyncio.sleep(10)
            yield None

    case = Case(id="h", language="en", login="none", channel="chat",
                turns=[{"user": "hi", "expect": {}}])
    result = await run_case(case, "m", FakeStore(), HangingLlm(), max_tool_rounds=6, run_id="r",
                            live={}, turn_timeout_s=0.2)
    assert not result.passed
    assert "timed out" in result.failures[0]
