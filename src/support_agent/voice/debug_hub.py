"""In-memory hub that streams debug events of voice calls."""

import asyncio
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from typing import Literal

MAX_CALLS = 20
QUEUE_SIZE = 200


class DebugHub:
    """Keeps the last calls in memory and streams their events.

    Each subscriber has a bounded queue. If a subscriber falls behind,
    the hub drops the oldest event of that queue. So a publisher never waits.
    """

    def __init__(self, max_calls: int = MAX_CALLS, queue_size: int = QUEUE_SIZE):
        self._calls: deque[dict] = deque(maxlen=max_calls)
        self._queue_size = queue_size
        self._subscribers: set[asyncio.Queue] = set()

    def start_call(self, channel: Literal["browser", "phone"], caller: str) -> str:
        call_id = uuid.uuid4().hex[:12]
        self._calls.append({
            "call_id": call_id,
            "channel": channel,
            "caller": caller,
            "started_at": time.time(),
            "ended_at": None,
            "events": [],
        })
        self.publish(call_id, "call_start", {"channel": channel, "caller": caller})
        return call_id

    def publish(self, call_id: str, kind: str, data: dict) -> None:
        call = self._find(call_id)
        if call is None:
            return
        event = {"call_id": call_id, "kind": kind, "data": data, "at": time.time()}
        call["events"].append(event)
        for queue in self._subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)

    def end_call(self, call_id: str) -> None:
        call = self._find(call_id)
        if call is None or call["ended_at"] is not None:
            return
        call["ended_at"] = time.time()
        self.publish(call_id, "call_end", {})

    def calls(self) -> list[dict]:
        return list(self._calls)

    def subscribe(self) -> AsyncIterator[dict]:
        """Register a subscriber now and return its event stream."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        return self._stream(queue)

    async def _stream(self, queue: asyncio.Queue) -> AsyncIterator[dict]:
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    def _find(self, call_id: str) -> dict | None:
        for call in self._calls:
            if call["call_id"] == call_id:
                return call
        return None
