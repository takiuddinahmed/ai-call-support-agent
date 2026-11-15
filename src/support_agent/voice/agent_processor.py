"""The Pipecat processor that runs one agent turn for each caller sentence.

Rules from the voice design ("Agent step", "Interruptions", "Order approval by voice"):

- Each final transcription starts one turn of run_turn(). The turn runs as a task, so that
  process_frame never waits for the model.
- Only one turn runs at a time. A transcription that comes during a turn, or during the
  answer audio of a turn, waits in a queue. It starts the next turn after the turn task
  ended and the answer audio ended.
- At the first tool call of a turn, the processor speaks a filler phrase. If answer text
  already went out in this turn, it does not speak the filler.
- The audio of each filler gives one BotStartedSpeakingFrame and BotStoppedSpeakingFrame
  pair. The answer audio is the first pair after the first LLMTextFrame of the turn that
  is not a filler pair.
- An InterruptionFrame before the answer audio does not stop the turn. The processor does
  not pass it on, so the agent audio continues. The caller speech becomes the next turn.
- An InterruptionFrame after the answer audio starts cancels the turn and goes on to the
  next processors, which stop the agent audio. run_turn keeps the history valid.
- After an interruption, the answer message in the history keeps only the text that the
  text-to-speech step spoke. SpokenTextTracker reports that text.
- The BotStoppedSpeakingFrame that ends the answer audio of a finished turn, with no
  interruption after the answer audio started, marks the order summaries of that turn as
  heard.
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
    TTSTextFrame,
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

# If the answer audio does not end in this time after the turn task ended, and the bot is
# silent, the processor stops the wait for that audio. The summary is then not heard.
ANSWER_AUDIO_TIMEOUT_S = 10.0

EventCallback = Callable[[str, dict], Awaitable[None]]


def _visible_length(text: str) -> int:
    """Counts the characters that are not white space."""
    return sum(1 for char in text if not char.isspace())


def _prefix_with_visible_length(text: str, count: int) -> str:
    """Returns the start of text that has count characters that are not white space."""
    if count <= 0:
        return ""
    seen = 0
    for index, char in enumerate(text):
        if not char.isspace():
            seen += 1
            if seen == count:
                return text[: index + 1]
    return text


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
        answer_audio_timeout_s: float = ANSWER_AUDIO_TIMEOUT_S,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session = session
        self._llm = llm
        self._runner = runner
        self._model = model
        self._max_tool_rounds = max_tool_rounds
        self._on_event = on_event
        self._answer_audio_timeout_s = answer_audio_timeout_s

        # Caller sentences that wait for the current turn and its answer audio to end.
        self._held: deque[str] = deque()
        self._turn_task: asyncio.Task | None = None
        self._audio_wait_task: asyncio.Task | None = None
        self._closing = False
        # True after EndFrame. Then the held turns do not wait for audio.
        self._ending = False
        self._filler_index = 0
        # True when a SpokenTextTracker reports the spoken text.
        self._tracks_spoken_text = False

        # Audio state. It is not reset at the start of a turn.
        self._bot_speaking = False
        # The number of filler audio pairs that did not start.
        self._filler_pairs_due = 0
        # True while the current audio pair is a filler pair.
        self._pair_is_filler = False

        self._reset_turn_state()

    def _reset_turn_state(self) -> None:
        # The customer turn number of the current turn, from session.customer_turns.
        self._turn_no: int | None = None
        # The index in session.messages where the messages of this turn start.
        self._first_message = len(self._session.messages)
        # True after the first LLMTextFrame of the turn went out.
        self._answer_text_sent = False
        # True after the answer audio of the turn started.
        self._bot_spoke_this_turn = False
        self._filler_sent = False
        # True after an InterruptionFrame that came after the answer audio started.
        self._interrupted = False
        # True after the turn gave its answer event.
        self._answer_done = False
        # True after a cancel of the turn. Then the turn task pushes no more frames.
        self._cancelled = False
        self._task_ended = False
        self._answer_audio_ended = False
        self._history_trimmed = False
        # All answer text of the turn that went out, and the visible length of the text
        # that went out before the last model round.
        self._sent_text = ""
        self._text_before_last_round = 0
        # The answer text that the text-to-speech step spoke in this turn.
        self._spoken: list[str] = []

    # Frames

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame, direction)
            return

        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self._handle_transcription(frame.text)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._handle_bot_started()
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped()
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

    def attach_spoken_text_tracker(self) -> None:
        """Turns on the history rule for the heard part of an interrupted answer."""
        self._tracks_spoken_text = True

    def note_spoken(self, text: str) -> None:
        """Records answer text that the text-to-speech step spoke."""
        # After an interruption, the heard part of the answer does not change.
        if not self._interrupted:
            self._spoken.append(text)

    def _handle_transcription(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._held.append(text)
        self._start_next_turn()

    def _handle_bot_started(self) -> None:
        if self._bot_speaking:
            return
        self._bot_speaking = True
        self._pair_is_filler = self._filler_pairs_due > 0
        if not self._pair_is_filler and self._answer_text_sent:
            self._bot_spoke_this_turn = True

    async def _handle_bot_stopped(self) -> None:
        if not self._bot_speaking:
            return
        self._bot_speaking = False
        if self._pair_is_filler:
            self._pair_is_filler = False
            self._filler_pairs_due -= 1
            return
        # The bot can also stop during the answer, for example while the model is slow.
        # The answer audio ends only after the turn task ended.
        if self._bot_spoke_this_turn and self._task_ended and not self._answer_audio_ended:
            self._answer_audio_ended = True
            await self._maybe_mark_heard()
            self._start_next_turn()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        if not self._bot_spoke_this_turn or self._answer_audio_ended:
            # The caller spoke before the answer audio. The turn continues and the next
            # transcription waits for it. The frame stops here, so the agent audio continues.
            return
        self._interrupted = True
        self._answer_audio_ended = True
        # The interruption stops all queued agent audio, also a filler that did not play.
        self._bot_speaking = False
        self._pair_is_filler = False
        self._filler_pairs_due = 0
        if self._turn_task is not None:
            if not self._cancelled:
                self._cancelled = True
                # The cancel does not wait for the task. A running tool can take seconds,
                # and run_turn lets it finish. The next turn starts when the task ends.
                self._turn_task.cancel()
        else:
            self._keep_heard_answer()
        # The base class resets the frame queue of this processor.
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        self._start_next_turn()

    # Turns

    def _answer_audio_settled(self) -> bool:
        if self._ending:
            return True
        return not self._answer_text_sent or self._answer_audio_ended

    def _start_next_turn(self) -> None:
        if self._closing or self._turn_task is not None or not self._held:
            return
        if not self._answer_audio_settled():
            return
        text = self._held.popleft()
        self._cancel_audio_wait()
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
            self._task_ended = True
            if self._interrupted:
                self._keep_heard_answer()
            elif not self._answer_audio_settled() and not self._closing:
                self._audio_wait_task = self.create_task(
                    self._wait_for_answer_audio(), name="agent_answer_audio_wait"
                )
            self._start_next_turn()

    async def _handle_event(self, event: AgentEvent) -> None:
        if event.type == "turn_start":
            # run_turn adds one to customer_turns before this event.
            self._turn_no = self._session.customer_turns
        elif event.type == "llm_start":
            self._text_before_last_round = _visible_length(self._sent_text)
        elif event.type == "text" and not self._cancelled:
            self._answer_text_sent = True
            self._sent_text += event.data["text"]
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
        self._filler_pairs_due += 1
        phrase = FILLER_PHRASES[self._filler_index % len(FILLER_PHRASES)]
        self._filler_index += 1
        # No context aggregator follows this processor. run_turn keeps the history.
        # SpokenTextTracker ignores the filler text, because append_to_context is False.
        await self.push_frame(TTSSpeakFrame(phrase, append_to_context=False))
        await self._emit("filler", {"text": phrase})

    async def _wait_for_answer_audio(self) -> None:
        """Stops the wait for answer audio that does not come, so that held turns run."""
        while not self._answer_audio_settled():
            await asyncio.sleep(self._answer_audio_timeout_s)
            if not self._bot_speaking and not self._answer_audio_settled():
                logger.warning(f"{self}: no end of the answer audio, the next turn starts")
                self._answer_audio_ended = True
        self._audio_wait_task = None
        self._start_next_turn()

    def _cancel_audio_wait(self) -> None:
        task = self._audio_wait_task
        self._audio_wait_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    # History and summaries

    def _keep_heard_answer(self) -> None:
        """Keeps only the spoken part of the interrupted answer in the history.

        The answer message is the last assistant message of the turn without tool calls.
        The rule does not change messages with tool calls, so each tool call keeps its
        tool result.
        """
        if not self._tracks_spoken_text or self._history_trimmed:
            return
        self._history_trimmed = True
        messages = self._session.messages
        index = next(
            (i for i in range(len(messages) - 1, self._first_message - 1, -1)
             if messages[i]["role"] == "assistant" and not messages[i].get("tool_calls")),
            None,
        )
        if index is None:
            return
        answer = messages[index]["content"] or ""
        spoken = _visible_length("".join(self._spoken)) - self._text_before_last_round
        heard = _prefix_with_visible_length(answer, spoken).strip()
        if heard:
            messages[index] = {**messages[index], "content": heard}
        else:
            # The caller heard nothing of the answer. A plain answer message has no tool
            # calls, so its removal keeps the history valid.
            del messages[index]

    async def _maybe_mark_heard(self) -> None:
        if not self._answer_done or self._interrupted or self._turn_no is None:
            return
        ids = [s.id for s in self._session.summaries.values()
               if s.customer_turn_at_creation == self._turn_no]
        self._session.mark_summaries_heard(self._turn_no)
        await self._emit("heard", {"turn": self._turn_no, "summary_ids": ids})

    async def _emit(self, kind: str, data: dict) -> None:
        try:
            await self._on_event(kind, data)
        except Exception:  # noqa: BLE001  a failed debug callback must not stop the call
            logger.exception(f"{self}: the event callback failed for {kind}")

    # End of the call

    async def _finish_turns(self) -> None:
        """Waits for the running turn and the held turns. EndFrame means a normal end."""
        self._ending = True
        self._cancel_audio_wait()
        self._start_next_turn()
        while (task := self._turn_task) is not None:
            await asyncio.wait([task])

    async def _stop_turns(self) -> None:
        self._closing = True
        self._held.clear()
        self._cancel_audio_wait()
        if (task := self._turn_task) is not None:
            self._cancelled = True
            await self.cancel_task(task)


class SpokenTextTracker(FrameProcessor):
    """Reports the answer text that the text-to-speech step spoke to an AgentProcessor.

    Put it after the TTS service. After the output transport, each TTSTextFrame comes after
    the audio of its text went to the transport. The tracker passes on all frames.
    """

    def __init__(self, agent: AgentProcessor, **kwargs):
        super().__init__(**kwargs)
        self._agent = agent
        agent.attach_spoken_text_tracker()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # A TTSSpeakFrame with append_to_context=False (the filler) is not answer text.
        if isinstance(frame, TTSTextFrame) and frame.append_to_context:
            self._agent.note_spoken(frame.text)
        await self.push_frame(frame, direction)
