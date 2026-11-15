from pipecat.frames.frames import (
    InterruptionFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from support_agent.session import Identity, Session
from support_agent.voice.agent_processor import AgentProcessor, SpokenTextTracker
from support_agent.voice.pipeline import TurnTextGate
from tests.unit.fakes import FakeLlm


def text(value, finalized=True):
    return TranscriptionFrame(value, "caller", "now", finalized=finalized)


def texts(frames):
    return [f.text for f in frames if isinstance(f, TranscriptionFrame)]


class Recorder:
    def __init__(self):
        self.events = []

    async def __call__(self, kind, data):
        self.events.append((kind, data))


async def test_gate_joins_the_segments_of_one_turn():
    events = Recorder()
    gate = TurnTextGate(on_event=events)
    down, _ = await run_test(gate, frames_to_send=[
        VADUserStartedSpeakingFrame(), SleepFrame(0.05), text(" I want", finalized=False),
        VADUserStartedSpeakingFrame(), SleepFrame(0.05), text(" an ESP32 board. "),
        SleepFrame(0.05), UserStoppedSpeakingFrame(), SleepFrame(0.05),
    ])
    assert texts(down) == ["I want an ESP32 board."]
    assert events.events == [("user_text", {"text": "I want an ESP32 board."})]


async def test_gate_waits_for_the_last_segment_after_the_turn_end():
    gate = TurnTextGate(on_event=Recorder())
    down, _ = await run_test(gate, frames_to_send=[
        VADUserStartedSpeakingFrame(), SleepFrame(0.05), UserStoppedSpeakingFrame(),
        SleepFrame(0.1), text("What is the price?"), SleepFrame(0.05),
    ])
    assert texts(down) == ["What is the price?"]


async def test_gate_sends_the_text_when_the_last_segment_gives_no_text():
    gate = TurnTextGate(on_event=Recorder(), segment_timeout_s=0.1)
    down, _ = await run_test(gate, frames_to_send=[
        VADUserStartedSpeakingFrame(), SleepFrame(0.05), text("Hello there", finalized=False),
        VADUserStartedSpeakingFrame(), SleepFrame(0.05), UserStoppedSpeakingFrame(),
        SleepFrame(0.3),
    ])
    assert texts(down) == ["Hello there"]


async def test_gate_keeps_the_text_through_an_interruption():
    gate = TurnTextGate(on_event=Recorder())
    down, _ = await run_test(gate, frames_to_send=[
        VADUserStartedSpeakingFrame(), SleepFrame(0.05), text("Is it in stock"),
        SleepFrame(0.05), InterruptionFrame(), SleepFrame(0.05),
        UserStoppedSpeakingFrame(), SleepFrame(0.05),
    ])
    assert texts(down) == ["Is it in stock"]


async def test_tracker_reports_the_spoken_answer_text():
    events = Recorder()
    agent = AgentProcessor(session=Session.new(Identity.guest(), "voice"), llm=FakeLlm([]),
                           runner=None, model="m", max_tool_rounds=6, on_event=events)
    tracker = SpokenTextTracker(agent, on_event=events)
    filler = TTSTextFrame("Let me check that.", text_type="sentence")
    filler.append_to_context = False
    await run_test(tracker, frames_to_send=[
        filler, TTSTextFrame("It costs five hundred taka.", text_type="sentence"),
    ])
    assert events.events == [("spoken", {"text": "It costs five hundred taka."})]
