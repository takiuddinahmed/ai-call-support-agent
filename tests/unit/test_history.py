import itertools
import json

from support_agent.history import (
    OLD_TOOL_RESULT_NOTE,
    HistoryWindow,
    estimate_tokens,
    fit_history,
    messages_tokens,
)


def tool_turn(n: int, result: str = "x" * 1500) -> list[dict]:
    """One customer turn with one tool call, its result, and an answer."""
    call_id = f"call_{n}"
    return [
        {"role": "user", "content": f"Question {n}"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": "search_products", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
        {"role": "assistant", "content": f"Answer {n} " + "y" * 200},
    ]


def text_turn(n: int) -> list[dict]:
    return [{"role": "user", "content": f"Question {n}"},
            {"role": "assistant", "content": f"Answer {n} " + "y" * 200}]


def assert_pairs_whole(messages: list[dict]) -> None:
    call_ids = [c["id"] for m in messages if m["role"] == "assistant"
                for c in m.get("tool_calls") or []]
    tool_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert sorted(call_ids) == sorted(tool_ids)


def test_estimate_counts_ascii_and_other_characters():
    assert estimate_tokens("a" * 300) == 100
    # Bangla text has fewer characters for each token.
    assert estimate_tokens("আ" * 300) == 200
    assert estimate_tokens("") == 0


def test_history_under_the_budget_stays_the_same():
    messages = text_turn(1) + text_turn(2)
    window = HistoryWindow()
    view = fit_history(messages, window, fixed_tokens=100, max_tokens=10_000)
    assert view == messages
    assert window == HistoryWindow()


def test_old_tool_results_shrink_first():
    messages = tool_turn(1) + tool_turn(2) + [{"role": "user", "content": "Question 3"}]
    fixed = 100
    budget = fixed + messages_tokens(messages) - 200
    view = fit_history(messages, HistoryWindow(), fixed_tokens=fixed, max_tokens=budget)
    tools = [m for m in view if m["role"] == "tool"]
    assert [m["content"] for m in tools] == [OLD_TOOL_RESULT_NOTE, OLD_TOOL_RESULT_NOTE]
    # No whole turn is dropped, because the shrink was enough.
    assert view[0] == messages[0]
    assert_pairs_whole(view)


def test_tool_results_of_the_current_turn_stay():
    messages = tool_turn(1) + tool_turn(2)[:3]
    view = fit_history(messages, HistoryWindow(), fixed_tokens=0, max_tokens=700)
    assert view[-1]["content"] == "x" * 1500
    assert view[-3] == {"role": "user", "content": "Question 2"}
    assert_pairs_whole(view)


def test_thirty_turn_conversation_stays_under_the_budget():
    window = HistoryWindow()
    messages: list[dict] = []
    fixed, budget = 2000, 3200
    for n in range(1, 31):
        messages += tool_turn(n) if n % 2 else text_turn(n)
        messages.append({"role": "user", "content": f"Next {n}"})
        view = fit_history(messages, window, fixed_tokens=fixed, max_tokens=budget)
        assert fixed + messages_tokens(view) <= budget
        assert view[0]["role"] == "user"
        assert view[-1] == messages[-1]
        assert_pairs_whole(view)
        messages.append({"role": "assistant", "content": "ok"})


def test_trim_leaves_room_so_the_next_turns_keep_the_same_start():
    window = HistoryWindow()
    messages: list[dict] = []
    fixed, budget = 1000, 2000
    starts = []
    for n in range(1, 21):
        messages += text_turn(n)
        fit_history(messages, window, fixed_tokens=fixed, max_tokens=budget)
        starts.append(window.start)
    changes = sum(1 for a, b in itertools.pairwise(starts) if a != b)
    # The trim goes below the budget, so it does not run on each turn.
    assert 0 < changes < 10


def test_latest_order_summary_is_kept():
    summary = json.dumps({"summary_id": "ab12cd34", "total_bdt": 1230})
    messages = tool_turn(1, result=summary)
    for n in range(2, 12):
        messages += text_turn(n)
    messages.append({"role": "user", "content": "yes"})
    view = fit_history(messages, HistoryWindow(), fixed_tokens=0, max_tokens=300,
                       keep_texts={"ab12cd34"})
    assert any(m["role"] == "tool" and m["content"] == summary for m in view)
    assert view[0] == messages[0]
    assert_pairs_whole(view)


def test_window_indexes_past_the_end_are_safe():
    messages = text_turn(1)
    window = HistoryWindow(start=40, compact_until=40)
    view = fit_history(messages, window, fixed_tokens=0, max_tokens=10_000)
    assert view == messages
