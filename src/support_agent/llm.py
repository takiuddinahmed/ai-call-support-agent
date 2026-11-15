"""Client for a local model server with the OpenAI-compatible chat API."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import httpx
from openai import AsyncOpenAI

from support_agent.config import Settings


@dataclass
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class LlmChunk:
    text: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    usage: dict | None = None


class LlmClient(Protocol):
    def stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LlmChunk]:
        ...


class OpenAiCompatClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    async def stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        reasoning_effort: str | None = None,
    ):
        # None uses the configuration value. An empty value sends nothing.
        effort = self.settings.llm_reasoning_effort if reasoning_effort is None else reasoning_effort
        extra = {}
        if effort:
            extra["reasoning_effort"] = effort
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or None,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.2,
            **extra,
        )
        async for chunk in response:
            usage = chunk.usage.model_dump() if chunk.usage else None
            if not chunk.choices:
                if usage:
                    yield LlmChunk(usage=usage)
                continue
            delta = chunk.choices[0].delta
            calls = None
            if delta.tool_calls:
                calls = [
                    ToolCallDelta(
                        index=c.index,
                        id=c.id,
                        name=c.function.name if c.function else None,
                        arguments=c.function.arguments if c.function else None,
                    )
                    for c in delta.tool_calls
                ]
            if delta.content or calls or usage:
                yield LlmChunk(text=delta.content or None, tool_calls=calls, usage=usage)

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10) as http:
            response = await http.get(f"{self.settings.llm_base_url}/models")
            response.raise_for_status()
            return sorted(m["id"] for m in response.json()["data"])
