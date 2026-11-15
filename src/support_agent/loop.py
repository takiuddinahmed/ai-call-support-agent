"""Runs one customer turn: model calls, tool calls, and timed events for the debug panel."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
import openai

from support_agent.history import (
    DEFAULT_MAX_PROMPT_TOKENS,
    estimate_tokens,
    fit_history,
    messages_tokens,
)
from support_agent.instructions import build_instructions
from support_agent.llm import LlmClient
from support_agent.session import Session
from support_agent.tools import TOOL_SPECS, ToolResult, ToolRunner

EventType = Literal[
    "turn_start", "llm_start", "first_token", "text", "tool_call_start", "llm_end",
    "tool_start", "tool_end", "answer", "error",
]


FINAL_ROUND_NOTE = (
    "Answer now with the information from the tool results above. Do not call tools. "
    "If something is missing, say so and ask the customer one short question."
)

# The tool result for a tool call that did not run because the customer interrupted the turn.
NOT_RUN_MESSAGE = "Not run: the customer interrupted."

# The agent says this line if a model call gives no text and no tool call.
EMPTY_ANSWER_FALLBACK = "Sorry, I did not get that. Can you say it again?"
EMPTY_ANSWER_ERROR = "The model gave an empty answer. The agent said the fallback line."


@dataclass
class AgentEvent:
    type: EventType
    t_ms: float
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "t_ms": round(self.t_ms, 1), "data": self.data},
                          ensure_ascii=False)


def _error_message(error: Exception) -> str:
    if isinstance(error, (httpx.ConnectError, openai.APIConnectionError)):
        return "The model server is not reachable. Start Ollama and try again."
    if isinstance(error, openai.NotFoundError):
        return "The model server does not have this model. Pull it or choose another model."
    return f"The model call failed: {error}"


def _add_tool_message(session: Session, call_id: str, content: str) -> None:
    session.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})


async def _wait_through_cancels(tool: asyncio.Future) -> ToolResult:
    """Waits until the tool task is done. Each further cancel during the wait has no effect."""
    while not tool.done():
        try:
            await asyncio.shield(tool)
        except asyncio.CancelledError:
            pass
    return tool.result()


async def run_turn(
    session: Session,
    user_text: str,
    llm: LlmClient,
    runner: ToolRunner,
    model: str,
    max_tool_rounds: int,
    reasoning_effort_after_tools: str = "",
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> AsyncIterator[AgentEvent]:
    """Streams one customer turn and yields timed events.

    The first model call of the turn uses the configured reasoning effort. If
    reasoning_effort_after_tools is not empty, each model call after a tool result uses it.

    The prompt (system prompt, tool specs, and history) stays under max_prompt_tokens
    (estimated tokens). history.fit_history() shrinks the older history.

    If a model call gives no text and no tool call, the turn yields an "error" event and
    answers with EMPTY_ANSWER_FALLBACK, so that the customer does not get silence.

    Cancel safety: a cancel (asyncio.CancelledError) can stop the turn at any await.
    After a cancel, session.messages stays valid for the next turn:

    - A cancel during a model stream drops the partial text of that round.
    - A cancel during a tool waits for that tool and records its real result.
      Further cancels during this wait have no effect.
    - Each tool call that did not run gets the tool message NOT_RUN_MESSAGE.

    Callers must close the generator with contextlib.aclosing(run_turn(...)).
    Then the finally block runs at once after a cancel, and not later at garbage collection.
    """
    start = time.perf_counter()

    def event(kind: EventType, **data) -> AgentEvent:
        return AgentEvent(kind, (time.perf_counter() - start) * 1000, data)

    session.customer_turns += 1
    session.messages.append({"role": "user", "content": user_text})
    yield event("turn_start", model=model, channel=session.channel)

    for round_no in range(1, max_tool_rounds + 2):
        # After the last tool round, the model must answer from the results it has.
        final_round = round_no > max_tool_rounds
        system = build_instructions(
            session.channel, session.identity, datetime.now(ZoneInfo("Asia/Dhaka"))
        )
        tools = [] if final_round else TOOL_SPECS
        head = [{"role": "system", "content": system}]
        tail = [{"role": "system", "content": FINAL_ROUND_NOTE}] if final_round else []
        fixed = messages_tokens(head + tail) + (estimate_tokens(json.dumps(tools)) if tools else 0)
        history = fit_history(session.messages, session.history, fixed, max_prompt_tokens,
                              keep_texts=set(session.summaries))
        messages = [*head, *history, *tail]
        # The first call keeps the configuration value (None). Later calls follow tool results.
        effort = (reasoning_effort_after_tools or None) if round_no > 1 else None
        yield event("llm_start", round=round_no, final=final_round)
        llm_start = time.perf_counter()
        first_token_ms = None
        text_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage: dict | None = None
        started: set[int] = set()
        try:
            async for chunk in llm.stream(model, messages, tools, reasoning_effort=effort):
                if first_token_ms is None and (chunk.text or chunk.tool_calls):
                    first_token_ms = (time.perf_counter() - llm_start) * 1000
                    yield event("first_token", round=round_no, ms=round(first_token_ms, 1))
                if chunk.text:
                    text_parts.append(chunk.text)
                    yield event("text", text=chunk.text)
                for delta in chunk.tool_calls or []:
                    call = calls.setdefault(delta.index, {"id": None, "name": "", "arguments": ""})
                    call["id"] = delta.id or call["id"]
                    call["name"] += delta.name or ""
                    call["arguments"] += delta.arguments or ""
                    if call["name"] and delta.index not in started:
                        started.add(delta.index)
                        yield event("tool_call_start", name=call["name"], round=round_no)
                if chunk.usage:
                    usage = chunk.usage
        except Exception as error:  # noqa: BLE001  any model failure ends the turn with one event
            if session.messages[-1]["role"] == "user":
                session.messages.pop()
            yield event("error", message=_error_message(error))
            return

        llm_ms = (time.perf_counter() - llm_start) * 1000
        # Tokens include hidden thinking tokens, which come before the first visible token.
        # So the speed uses the whole model call time.
        tokens = (usage or {}).get("completion_tokens")
        gen_ms = llm_ms
        yield event(
            "llm_end",
            round=round_no,
            ms=round(llm_ms, 1),
            first_token_ms=round(first_token_ms or 0, 1),
            tokens=tokens,
            tokens_per_s=round(tokens / (gen_ms / 1000), 1) if tokens and gen_ms > 0 else None,
        )

        text = "".join(text_parts).strip()
        if not text and (not calls or final_round):
            # For example, the hidden thinking filled the model context.
            yield event("error", message=EMPTY_ANSWER_ERROR, round=round_no, fallback=True)
            # The customer gets only the fallback line, not an order summary from this turn.
            # So a later "yes" must not place that summary.
            for summary_id in [s.id for s in session.summaries.values()
                               if s.customer_turn_at_creation == session.customer_turns]:
                del session.summaries[summary_id]
            text = EMPTY_ANSWER_FALLBACK
            yield event("text", text=text)
        if not calls or final_round:
            session.messages.append({"role": "assistant", "content": text})
            yield event("answer", text=text, total_ms=round((time.perf_counter() - start) * 1000, 1))
            return

        ordered = [calls[i] for i in sorted(calls)]
        for i, call in enumerate(ordered):
            call["id"] = call["id"] or f"call_{round_no}_{i}"
        session.messages.append({
            "role": "assistant",
            "content": text or None,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for c in ordered
            ],
        })
        answered = 0
        try:
            for call in ordered:
                yield event("tool_start", name=call["name"], arguments=call["arguments"])
                # The shield lets a started tool finish after a cancel, so that its result
                # (for example a placed order) goes into the history.
                tool = asyncio.ensure_future(runner.run(call["name"], call["arguments"]))
                try:
                    result = await asyncio.shield(tool)
                except asyncio.CancelledError:
                    result = await _wait_through_cancels(tool)
                    _add_tool_message(session, call["id"], result.content)
                    answered += 1
                    raise
                _add_tool_message(session, call["id"], result.content)
                answered += 1
                yield event("tool_end", name=call["name"], ok=result.ok, content=result.content,
                            ms=round(result.ms, 1))
        finally:
            # If the turn stops early, the calls that did not run get a tool message too.
            for call in ordered[answered:]:
                _add_tool_message(session, call["id"], NOT_RUN_MESSAGE)
