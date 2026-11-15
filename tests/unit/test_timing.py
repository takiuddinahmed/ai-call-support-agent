from pipecat.frames.frames import (
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

from support_agent.voice.timing import TurnTimer


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_timer():
    events = []

    async def on_event(kind, data):
        events.append((kind, data))

    clock = FakeClock()
    return TurnTimer(on_event, clock=clock), clock, events


async def push(timer, clock, at, frame):
    clock.now = at
    await timer.on_push_frame(
        FramePushed(source=None, destination=None, frame=frame,
                    direction=FrameDirection.DOWNSTREAM, timestamp=0)
    )


def audio():
    return TTSAudioRawFrame(audio=b"\0\0", sample_rate=24000, num_channels=1)


async def test_pause_measures_speech_end_to_first_audio():
    timer, clock, events = make_timer()
    await push(timer, clock, 0.0, VADUserStoppedSpeakingFrame())
    await push(timer, clock, 0.3, UserStoppedSpeakingFrame())
    await push(timer, clock, 0.6, TranscriptionFrame(text="hello", user_id="u", timestamp="t"))
    first = audio()
    await push(timer, clock, 1.4, first)
    await push(timer, clock, 1.5, first)  # same frame at the next hop
    await push(timer, clock, 1.6, audio())  # second chunk of the same turn
    assert events == [
        ("speech_end", {"ms": 300}),
        ("stt", {"ms": 300, "text": "hello"}),
        ("first_audio", {"ms": 1100}),
        ("pause", {"ms": 1400}),
    ]


async def test_next_turn_is_measured_again():
    timer, clock, events = make_timer()
    for base in (0.0, 10.0):
        await push(timer, clock, base, VADUserStoppedSpeakingFrame())
        await push(timer, clock, base + 0.2, UserStoppedSpeakingFrame())
        await push(timer, clock, base + 1.0, audio())
    pauses = [d["ms"] for k, d in events if k == "pause"]
    assert pauses == [1000, 1000]


async def test_audio_without_turn_is_ignored():
    timer, clock, events = make_timer()
    await push(timer, clock, 1.0, audio())
    assert events == []
