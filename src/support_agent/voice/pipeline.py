"""The Pipecat pipeline of one voice call, and the voice models that all calls share.

The pipeline order, from the voice design ("Pipeline"):

1. transport.input(): the caller audio.
2. VADProcessor with Silero: finds speech. UserTurnProcessor needs these VAD frames.
3. UserTurnProcessor: a VAD start strategy and a smart-turn stop strategy.
4. WhisperSTTServiceMLX: English speech-to-text on the Apple GPU.
5. TurnTextGate: joins the text of one caller turn.
6. AgentProcessor: runs run_turn() for each caller turn.
7. KokoroTTSService: speaks the answer one sentence at a time.
8. transport.output(): the agent audio.
9. SpokenTextTracker: reports the answer text that went out as audio.
"""

import asyncio

from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.tts_service import TTSService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601

from support_agent.session import Session
from support_agent.tools import ToolRunner
from support_agent.voice.agent_processor import AgentProcessor, EventCallback, SpokenTextTracker
from support_agent.voice.debug_hub import DebugHub
from support_agent.voice.deps import VoiceDeps
from support_agent.voice.timing import TurnTimer

# If the last speech segment of a turn gives no text in this time after the turn end,
# the gate sends the text that it has.
SEGMENT_TIMEOUT_S = 3.0


class TurnTextGate(FrameProcessor):
    """Joins the transcriptions of one caller turn into one TranscriptionFrame.

    The segmented Whisper service transcribes each VAD speech segment. A short pause in
    a sentence gives two segments, but the smart-turn model can decide that the turn
    continues. The AgentProcessor starts one agent turn for each TranscriptionFrame. So
    this gate holds the text until the turn ends (UserStoppedSpeakingFrame) and the
    text of the last segment came.

    A segment is pending from its VADUserStartedSpeakingFrame to a finalized
    TranscriptionFrame. If Whisper gives no text for the last segment, the gate sends
    the text that it has `segment_timeout_s` after the turn end.

    The gate does not run the base interruption handler, so that a queued
    transcription stays when the caller starts to speak again.
    """

    def __init__(self, on_event: EventCallback, segment_timeout_s: float = SEGMENT_TIMEOUT_S,
                 **kwargs):
        super().__init__(**kwargs)
        self._on_event = on_event
        self._segment_timeout_s = segment_timeout_s
        self._parts: list[str] = []
        self._user_id = ""
        self._segment_pending = False
        self._turn_ended = False
        self._timeout_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            return
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                self._parts.append(frame.text.strip())
                self._user_id = frame.user_id
            if frame.finalized:
                self._segment_pending = False
                if self._turn_ended:
                    await self._send()
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._segment_pending = True
        elif isinstance(frame, EndFrame) and self._turn_ended:
            await self._send()
        await self.push_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_ended = True
            if self._segment_pending:
                self._timeout_task = self.create_task(self._send_after_timeout(),
                                                      name="turn_text_timeout")
            else:
                await self._send()

    async def cleanup(self):
        await self._cancel_timeout()
        await super().cleanup()

    async def _send_after_timeout(self) -> None:
        await asyncio.sleep(self._segment_timeout_s)
        self._timeout_task = None
        logger.warning(f"{self}: no text for the last speech segment, the turn text goes on")
        await self._send()

    async def _send(self) -> None:
        await self._cancel_timeout()
        text = " ".join(self._parts)
        self._parts = []
        self._turn_ended = False
        self._segment_pending = False
        if not text:
            return
        try:
            await self._on_event("user_text", {"text": text})
        except Exception:  # noqa: BLE001  a failed debug callback must not stop the call
            logger.exception(f"{self}: the event callback failed for user_text")
        await self.push_frame(TranscriptionFrame(text, self._user_id, time_now_iso8601(),
                                                 finalized=True))

    async def _cancel_timeout(self) -> None:
        task = self._timeout_task
        self._timeout_task = None
        if task is not None and task is not asyncio.current_task():
            await self.cancel_task(task)


class SharedKokoroTTSService(KokoroTTSService):
    """KokoroTTSService that uses a Kokoro model that is already loaded.

    KokoroTTSService.__init__ loads the model files each time. This class calls the
    TTSService constructor with the same arguments and keeps the shared model, so that
    each call does not load the model again.
    """

    def __init__(self, *, kokoro, voice: str, **kwargs):
        TTSService.__init__(
            self,
            push_start_frame=True,
            push_stop_frames=True,
            settings=self.Settings(model=None, voice=voice, language=Language.EN, speed=1.0),
            **kwargs,
        )
        self._kokoro = kokoro
        if voice not in kokoro.voices:
            raise ValueError(f"Kokoro TTS voice '{voice}' is not in the voices file")


def build_task(transport: BaseTransport, session: Session, deps: VoiceDeps, hub: DebugHub,
               call_id: str, audio_rate: int) -> PipelineWorker:
    """Builds the pipeline of one call. The voice models must be loaded first."""
    if not deps.models.loaded:
        raise RuntimeError("Load the voice models before the call starts.")
    settings = deps.settings

    async def on_event(kind: str, data: dict) -> None:
        # The streamed text pieces are too many for the panel. The "answer" event has the
        # full text, and "spoken" has the text that went out as audio.
        if kind != "text":
            hub.publish(call_id, kind, data)

    agent = AgentProcessor(
        session=session,
        llm=deps.llm,
        runner=ToolRunner(deps.store, session),
        model=settings.llm_model,
        max_tool_rounds=settings.max_tool_rounds,
        on_event=on_event,
        reasoning_effort_after_tools=settings.llm_reasoning_effort_after_tools,
        max_prompt_tokens=settings.max_prompt_tokens,
    )
    turns = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            # The STT service is after this processor, so no transcription comes here.
            # The stop strategy must not wait for one.
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3(),
                                                   wait_for_transcript=False)],
        ),
    )
    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(model=settings.voice_stt_model,
                                               language=Language.EN),
    )
    tts = SharedKokoroTTSService(kokoro=deps.models.kokoro, voice=settings.voice_tts_voice)
    pipeline = Pipeline([
        transport.input(),
        VADProcessor(vad_analyzer=SileroVADAnalyzer()),
        turns,
        stt,
        TurnTextGate(on_event=on_event),
        agent,
        tts,
        transport.output(),
        # The tracker is after the output transport. There, each TTSTextFrame comes
        # after its audio went to the transport, so the spoken text is closer to the
        # text that the caller heard.
        SpokenTextTracker(agent, on_event=on_event),
    ])
    return PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=audio_rate, audio_out_sample_rate=audio_rate),
        observers=[TurnTimer(on_event)],
        enable_rtvi=False,
    )
