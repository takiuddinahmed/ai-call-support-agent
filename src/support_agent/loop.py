"""Runs one customer turn: model calls, tool calls, and timed events for the debug panel."""

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
import openai

from support_agent.instructions import build_instructions
from support_agent.llm import LlmClient
from support_agent.session import Session
from support_agent.tools import TOOL_SPECS, ToolRunner

EventType = Literal[
    "turn_start", "llm_start", "first_token", "text", "llm_end",
    "tool_start", "tool_end", "answer", "error",
]


FINAL_ROUND_NOTE = (
    "Answer now with the information from the tool results above. Do not call tools. "
    "If something is missing, say so and ask the customer one short question."
)


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


async def run_turn(
    session: Session,
    user_text: str,
    llm: LlmClient,
    runner: ToolRunner,
    model: str,
    max_tool_rounds: int,
) -> AsyncIterator[AgentEvent]:
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
        messages = [{"role": "system", "content": system}, *session.messages]
        if final_round:
            messages.append({"role": "system", "content": FINAL_ROUND_NOTE})
        yield event("llm_start", round=round_no, final=final_round)
        llm_start = time.perf_counter()
        first_token_ms = None
        text_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage: dict | None = None
        try:
            async for chunk in llm.stream(model, messages, [] if final_round else TOOL_SPECS):
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
        if final_round and not text:
            yield event("error", message="The agent could not finish the answer. Ask again.")
            return
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
        for call in ordered:
            yield event("tool_start", name=call["name"], arguments=call["arguments"])
            result = await runner.run(call["name"], call["arguments"])
            session.messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result.content}
            )
            yield event("tool_end", name=call["name"], ok=result.ok, content=result.content,
                        ms=round(result.ms, 1))
