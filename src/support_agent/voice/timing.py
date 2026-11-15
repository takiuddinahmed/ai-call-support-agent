"""Pause measurement for voice turns, as a Pipecat observer."""

import time
from collections.abc import Awaitable, Callable

from pipecat.frames.frames import (
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

EventCallback = Callable[[str, dict], Awaitable[None]]


class TurnTimer(BaseObserver):
    """Measures the pause of each turn and reports it through `on_event`.

    An observer sees each frame at every hop, so the timer counts each frame
    once by its frame ID and counts each event once for each turn.
    The timer reads its clock when the observer receives the frame.

    Events, with `ms` in whole milliseconds:
    - "speech_end": VADUserStoppedSpeakingFrame to UserStoppedSpeakingFrame.
    - "stt": the turn-end decision to the first TranscriptionFrame, with `text`.
      If the transcription comes before the turn-end decision, the timer
      measures from the VAD stop instead.
    - "first_audio": the turn-end decision to the first TTSAudioRawFrame.
    - "pause": the VAD stop to the first TTSAudioRawFrame.

    The timer reports the first TTSAudioRawFrame that it sees in a turn.
    That frame is the one that the TTS service pushes, which is the first hop
    on the way to the output transport.
    """

    def __init__(self, on_event: EventCallback, clock: Callable[[], float] = time.perf_counter):
        super().__init__()
        self._on_event = on_event
        self._clock = clock
        self._vad_stop: float | None = None
        self._turn_end: float | None = None
        self._stt_done = False
        self._audio_done = False
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if isinstance(frame, TTSAudioRawFrame):
            await self._on_audio()
            return
        if not isinstance(
            frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame, TranscriptionFrame)
        ):
            return
        if frame.id in self._seen:
            return
        self._seen.add(frame.id)
        now = self._clock()
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_stop = now
            self._turn_end = None
            self._stt_done = False
            self._audio_done = False
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._vad_stop is not None and self._turn_end is None:
                self._turn_end = now
                await self._on_event("speech_end", {"ms": _ms(now - self._vad_stop)})
        elif not self._stt_done:
            start = self._turn_end if self._turn_end is not None else self._vad_stop
            if start is not None:
                self._stt_done = True
                await self._on_event("stt", {"ms": _ms(now - start), "text": frame.text})

    async def _on_audio(self) -> None:
        if self._audio_done or self._vad_stop is None:
            return
        self._audio_done = True
        now = self._clock()
        if self._turn_end is not None:
            await self._on_event("first_audio", {"ms": _ms(now - self._turn_end)})
        await self._on_event("pause", {"ms": _ms(now - self._vad_stop)})


def _ms(seconds: float) -> int:
    return round(seconds * 1000)
