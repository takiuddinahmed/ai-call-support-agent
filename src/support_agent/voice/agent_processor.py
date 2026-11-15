"""The Pipecat processor that runs one agent turn for each caller sentence.

Rules from the voice design ("Agent step", "Interruptions", "Order approval by voice"):

- Each final transcription starts one turn of run_turn(). The turn runs as a task, so that
  process_frame never waits for the model.
- Only one turn runs at a time. A transcription that comes during a turn waits in a queue
  and starts the next turn after the current turn ends.
- At the first tool call of a turn, the processor speaks a filler phrase. If answer text
  already went out in this turn, it does not speak the filler.
- Answer audio starts at the first BotStartedSpeakingFrame after the first LLMTextFrame of
  the turn. The filler audio is not answer audio.
- An InterruptionFrame before the answer audio does not stop the turn. The processor does
  not pass it on, so the agent audio continues. The caller speech becomes the next turn.
- An InterruptionFrame after the answer audio starts cancels the turn and goes on to the
  next processors, which stop the agent audio. run_turn keeps the history valid.
- The BotStoppedSpeakingFrame after the answer of a finished turn, with no interruption
  after the answer audio started, marks the order summaries of that turn as heard.
"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import aclosing

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from support_agent.llm import LlmClient
from support_agent.loop import AgentEvent, run_turn
from support_agent.session import Session
from support_agent.tools import ToolRunner

# Customer-facing phrases. The processor uses them in turn, so that two tool calls in a row
# do not give the same phrase.
FILLER_PHRASES: list[str] = [
    "Let me check that.",
    "One moment, please.",
    "Let me look that up.",
    "Please wait a moment.",
]

EventCallback = Callable[[str, dict], Awaitable[None]]


class AgentProcessor(FrameProcessor):
    """Runs run_turn() for each caller transcription and streams the answer text to TTS."""

    def __init__(
        self,
        session: Session,
        llm: LlmClient,
        runner: ToolRunner,
        model: str,
        max_tool_rounds: int,
        on_event: EventCallback,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session = session
        self._llm = llm
        self._runner = runner
        self._model = model
        self._max_tool_rounds = max_tool_rounds
        self._on_event = on_event

        # Caller sentences that wait for the current turn to end.
        self._held: deque[str] = deque()
        self._turn_task: asyncio.Task | None = None
        self._closing = False
        self._filler_index = 0
        self._reset_turn_state()

    def _reset_turn_state(self) -> None:
        # The customer turn number of the current turn, from session.customer_turns.
        self._turn_no: int | None = None
        # True after the first LLMTextFrame of the turn went out.
        self._answer_text_sent = False
        # True after the first BotStartedSpeakingFrame that follows the first answer text.
        self._bot_spoke_this_turn = False
        self._filler_sent = False
        # True after an InterruptionFrame that came after the answer audio started.
        self._interrupted = False
        # True after the turn gave its answer event. Then the next stop of the answer audio
        # can mark the summaries as heard.
        self._answer_done = False
        # True after a cancel of the turn. Then the turn task pushes no more frames.
        self._cancelled = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame, direction)
            return

        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self._handle_transcription(frame.text)
        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._answer_text_sent:
                self._bot_spoke_this_turn = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._maybe_mark_heard()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._finish_turns()
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self._stop_turns()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        await self._stop_turns()
        await super().cleanup()

    def _handle_transcription(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._held.append(text)
        self._start_next_turn()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        if not self._bot_spoke_this_turn:
            # The caller spoke before the answer audio. The turn continues and the next
            # transcription waits for it. The frame stops here, so the agent audio continues.
            return
        self._interrupted = True
        if self._turn_task is not None and not self._cancelled:
            self._cancelled = True
            # The cancel does not wait for the task. A running tool can take seconds, and
            # run_turn lets it finish. The next turn starts when the task ends.
            self._turn_task.cancel()
        # The base class resets the frame queue of this processor.
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    def _start_next_turn(self) -> None:
        if self._closing or self._turn_task is not None or not self._held:
            return
        text = self._held.popleft()
        self._reset_turn_state()
        self._turn_task = self.create_task(self._run_turn(text), name="agent_turn")

    async def _run_turn(self, text: str) -> None:
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            async with aclosing(run_turn(
                self._session, text, self._llm, self._runner, self._model,
                self._max_tool_rounds,
            )) as events:
                async for event in events:
                    await self._handle_event(event)
            if not self._cancelled:
                await self.push_frame(LLMFullResponseEndFrame())
        finally:
            self._turn_task = None
            self._start_next_turn()

    async def _handle_event(self, event: AgentEvent) -> None:
        if event.type == "turn_start":
            # run_turn adds one to customer_turns before this event.
            self._turn_no = self._session.customer_turns
        elif event.type == "text" and not self._cancelled:
            self._answer_text_sent = True
            await self.push_frame(LLMTextFrame(event.data["text"]))
        elif event.type == "tool_call_start" and not self._answer_text_sent:
            await self._speak_filler()
        elif event.type == "answer":
            self._answer_done = True
        await self._emit(event.type, {**event.data, "t_ms": round(event.t_ms, 1)})

    async def _speak_filler(self) -> None:
        if self._filler_sent or self._cancelled:
            return
        self._filler_sent = True
        phrase = FILLER_PHRASES[self._filler_index % len(FILLER_PHRASES)]
        self._filler_index += 1
        # No context aggregator follows this processor. run_turn keeps the history.
        await self.push_frame(TTSSpeakFrame(phrase, append_to_context=False))
        await self._emit("filler", {"text": phrase})

    async def _maybe_mark_heard(self) -> None:
        if not (self._answer_done and self._bot_spoke_this_turn) or self._interrupted:
            return
        if self._turn_no is None:
            return
        self._answer_done = False
        ids = [s.id for s in self._session.summaries.values()
               if s.customer_turn_at_creation == self._turn_no]
        self._session.mark_summaries_heard(self._turn_no)
        await self._emit("heard", {"turn": self._turn_no, "summary_ids": ids})

    async def _emit(self, kind: str, data: dict) -> None:
        try:
            await self._on_event(kind, data)
        except Exception:  # noqa: BLE001  a failed debug callback must not stop the call
            logger.exception(f"{self}: the event callback failed for {kind}")

    async def _finish_turns(self) -> None:
        """Waits for the running turn and the held turns. EndFrame means a normal end."""
        while (task := self._turn_task) is not None:
            await asyncio.wait([task])

    async def _stop_turns(self) -> None:
        self._closing = True
        self._held.clear()
        if (task := self._turn_task) is not None:
            self._cancelled = True
            await self.cancel_task(task)
