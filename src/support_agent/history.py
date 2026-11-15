"""Keeps the prompt of a model call under a token budget.

The model server has a fixed context (Ollama: 4096 tokens by default). If the prompt fills
the context, no room stays for the hidden thinking and the answer, and the model gives an
empty answer. So run_turn() sends a bounded view of the history.

The view keeps the system prompt (the caller adds it) and the latest turns. If the prompt
is too large, the view shrinks in this order:

1. The tool results of older turns become OLD_TOOL_RESULT_NOTE.
2. The oldest whole customer turns go out of the view.

A turn is a customer message and all messages until the next customer message. So a tool
call and its tool result always stay together. The current turn always stays. A tool result
that contains one of `keep_texts` (the ID of the pending order summary) never shrinks, and
the trim stops before the turn that has it, because place_order needs that summary.

session.messages does not change. The voice processor keeps indexes into that list.

The trim goes down to the middle of the free room, and not only to the budget. Then the
start of the prompt stays the same for the next turns, and the model server can use its
prompt cache.
"""

import json
from dataclasses import dataclass

# A token estimate without the model tokenizer. Measured with qwen3.5:4b on 2026-11-15:
# the JSON of the agent messages and tools has 3.5 to 4.3 characters for each token, and
# Bangla text has about 1.9. The estimate uses 3 and 1.5, so it is a little too high.
ASCII_CHARS_PER_TOKEN = 3
OTHER_CHARS_PER_TOKEN = 1.5

# The default prompt budget, in estimated tokens. It is for the Ollama context of 4096
# tokens. The estimate is a little too high, so the real prompt is about 2700 tokens, and
# about 1400 tokens stay for the hidden thinking and the answer. config.py and loop.py
# use this value.
DEFAULT_MAX_PROMPT_TOKENS = 3200

OLD_TOOL_RESULT_NOTE = (
    "Old tool result removed to save space. Call the tool again if you need these facts."
)


@dataclass
class HistoryWindow:
    """The part of session.messages that the prompt shows. It moves only forward."""

    # The index of the first message in the view.
    start: int = 0
    # Tool results before this index show OLD_TOOL_RESULT_NOTE.
    compact_until: int = 0


def estimate_tokens(text: str) -> int:
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other_chars = len(text) - ascii_chars
    return round(ascii_chars / ASCII_CHARS_PER_TOKEN + other_chars / OTHER_CHARS_PER_TOKEN)


def messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)


def _protected(message: dict, keep_texts: set[str]) -> bool:
    content = message.get("content") or ""
    return message["role"] == "tool" and any(text in content for text in keep_texts)


def _view(messages: list[dict], window: HistoryWindow, keep_texts: set[str]) -> list[dict]:
    view = []
    for index in range(window.start, len(messages)):
        message = messages[index]
        if (index < window.compact_until and message["role"] == "tool"
                and not _protected(message, keep_texts)):
            message = {**message, "content": OLD_TOOL_RESULT_NOTE}
        view.append(message)
    return view


def fit_history(
    messages: list[dict],
    window: HistoryWindow,
    fixed_tokens: int,
    max_tokens: int,
    keep_texts: set[str] | None = None,
) -> list[dict]:
    """Returns the messages to send after the system prompt. Changes `window` if it trims.

    fixed_tokens is the estimate of the system prompt and the tool specs.
    """
    keep = keep_texts or set()
    if window.start > len(messages):
        # The history was replaced (for example by Session.start_over).
        window.start = window.compact_until = 0
    view = _view(messages, window, keep)
    if fixed_tokens + messages_tokens(view) <= max_tokens:
        return view

    target = fixed_tokens + max(0, max_tokens - fixed_tokens) // 2
    turn_starts = [i for i, m in enumerate(messages) if m["role"] == "user"]
    current = turn_starts[-1] if turn_starts else len(messages)

    # Step 1: shrink the tool results of all turns before the current turn.
    window.compact_until = max(window.compact_until, current)
    view = _view(messages, window, keep)
    if fixed_tokens + messages_tokens(view) <= target:
        return view

    # Step 2: drop the oldest whole turns, but never the current turn.
    for next_start in turn_starts:
        if next_start <= window.start:
            continue
        if next_start > current:
            break
        if any(_protected(m, keep) for m in messages[window.start:next_start]):
            break
        window.start = next_start
        view = _view(messages, window, keep)
        if fixed_tokens + messages_tokens(view) <= target:
            break
    return view
