import asyncio
import json
import uuid

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import SleepFrame, run_test

from support_agent.session import Identity, Session
from support_agent.tools import ToolRunner
from support_agent.voice.agent_processor import (
    FILLER_PHRASES,
    AgentProcessor,
    SpokenTextTracker,
)
from tests.unit.fakes import FakeLlm, FakeStore

ADDRESS = {"first_name": "Karim", "address_1": "Road 5, Mirpur", "city": "Dhaka",
           "phone": "01712345678"}
PREPARE = json.dumps({"items": [{"handle": "esp32-devkit-v1", "quantity": 1}],
                      "address": ADDRESS})
SUMMARY_TEXT = "One ESP32 board to Mirpur. The total is six hundred thirty taka. Say yes to order."


class MixedLlm(FakeLlm):
    """FakeLlm that also plays a round of (text, [(name, arguments)])."""

    async def stream(self, model, messages, tools, reasoning_effort=None):
        from support_agent.llm import LlmChunk, ToolCallDelta

        step = self.rounds[0]
        if not isinstance(step, tuple):
            async for chunk in super().stream(model, messages, tools, reasoning_effort):
                yield chunk
            return
        self.rounds.pop(0)
        self.seen.append(list(messages))
        text, calls = step
        yield LlmChunk(text=text)
        yield LlmChunk(tool_calls=[
            ToolCallDelta(index=i, id=f"call_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ])


class Setup:
    def __init__(self, llm, store=None):
        self.session = Session.new(Identity.guest(), "voice")
        self.store = store or FakeStore()
        self.llm = llm
        self.events: list[tuple[str, dict]] = []

        async def on_event(kind, data):
            self.events.append((kind, data))

        self.processor = AgentProcessor(
            session=self.session,
            llm=llm,
            runner=ToolRunner(self.store, self.session),
            model="m",
            max_tool_rounds=6,
            on_event=on_event,
        )

    def kinds(self):
        return [kind for kind, _ in self.events]

    async def run(self, frames, **kwargs):
        down, _ = await run_test(self.processor, frames_to_send=frames, **kwargs)
        return down

    async def run_with_tracker(self, frames):
        pipeline = Pipeline([self.processor, SpokenTextTracker(self.processor)])
        down, _ = await run_test(pipeline, frames_to_send=frames)
        return down


def said(text):
    return TranscriptionFrame(text=text, user_id="caller", timestamp="2026-11-15T10:00:00Z")


def spoken(text, append_to_context=True):
    frame = TTSTextFrame(text, text_type="sentence")
    frame.append_to_context = append_to_context
    return frame


def audio_pair():
    """One BotStartedSpeakingFrame and BotStoppedSpeakingFrame pair, as for one TTS context."""
    return [BotStartedSpeakingFrame(), SleepFrame(0.05), BotStoppedSpeakingFrame(), SleepFrame(0.05)]


def index_of(frames, frame_type, start=0):
    return next(i for i, f in enumerate(frames) if i >= start and isinstance(f, frame_type))


async def test_transcription_streams_answer_frames():
    setup = Setup(FakeLlm(["Hello there."]))
    down = await setup.run(
        [said("hi")],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMTextFrame,
                              LLMFullResponseEndFrame],
    )
    assert "".join(f.text for f in down if isinstance(f, LLMTextFrame)) == "Hello there. "
    assert setup.kinds()[0] == "turn_start" and setup.kinds()[-1] == "answer"
    assert all("t_ms" in data for _, data in setup.events)


async def test_filler_spoken_on_tool_call():
    setup = Setup(FakeLlm([[("search_products", '{"text":"esp32"}')], "It costs 550 taka."]))
    down = await setup.run([said("price of esp32")])
    kinds = [type(f) for f in down]
    assert TTSSpeakFrame in kinds
    assert kinds.index(TTSSpeakFrame) < kinds.index(LLMTextFrame)
    fillers = [f for f in down if isinstance(f, TTSSpeakFrame)]
    assert len(fillers) == 1 and fillers[0].text in FILLER_PHRASES
    assert ("filler", fillers[0].text) in [(k, d.get("text")) for k, d in setup.events]


async def test_no_filler_after_answer_text():
    llm = MixedLlm([("Sure.", [("search_products", '{"text":"esp32"}')]), "It costs 550 taka."])
    setup = Setup(llm)
    down = await setup.run([said("price of esp32")])
    assert not any(isinstance(f, TTSSpeakFrame) for f in down)
    assert "filler" not in setup.kinds()
    assert any(isinstance(f, LLMTextFrame) and f.text == "Sure." for f in down)


async def test_empty_transcription_is_ignored():
    llm = FakeLlm(["Hello."])
    setup = Setup(llm)
    down = await setup.run([said("  ")])
    assert llm.seen == []
    assert setup.session.customer_turns == 0
    assert down == []


async def test_words_before_the_answer_audio_replace_the_turn():
    # The caller says "hello" two times while the model is slow. The agent answers once.
    llm = FakeLlm(["Hello! How can I help you?", "Hello! How can I help you?"], delay_s=0.05)
    setup = Setup(llm)
    down = await setup.run([
        said("Hello."), SleepFrame(0.02),
        InterruptionFrame(), SleepFrame(0.02), said("Hello."), SleepFrame(0.6),
    ])
    assert setup.kinds().count("answer") == 1
    # The processor stops the first turn, and the text-to-speech step drops its text.
    assert sum(isinstance(f, InterruptionFrame) for f in down) == 1
    assert setup.session.messages == [
        {"role": "user", "content": "Hello."},
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]
    assert ("interrupted", "words") in [(k, d.get("by")) for k, d in setup.events]


async def test_words_during_the_filler_replace_the_turn():
    llm = FakeLlm([[("search_products", '{"text":"esp32"}')], "It costs 550 taka.",
                   "Sure, tell me the order number."], delay_s=0.05)
    setup = Setup(llm)
    down = await setup.run_with_tracker([
        said("price of esp32"), SleepFrame(0.08),
        BotStartedSpeakingFrame(), SleepFrame(0.02),  # the filler audio
        InterruptionFrame(), SleepFrame(0.02), said("no, track my order"), SleepFrame(0.8),
    ])
    assert any(isinstance(f, TTSSpeakFrame) for f in down)
    assert sum(isinstance(f, InterruptionFrame) for f in down) == 1
    answers = [d["text"].strip() for k, d in setup.events if k == "answer"]
    assert answers == ["Sure, tell me the order number."]
    # The unheard answer of turn 1 is not in the history. The tool call pair stays whole.
    roles = [m["role"] for m in setup.session.messages]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]


async def test_words_during_the_answer_audio_stop_it_without_a_vad_interruption():
    llm = FakeLlm(["It costs five hundred fifty taka for one board today.", "Okay."],
                  delay_s=0.1)
    setup = Setup(llm)
    down = await setup.run([
        said("price of esp32"), SleepFrame(0.25),
        BotStartedSpeakingFrame(), SleepFrame(0.05),
        said("stop"), SleepFrame(0.5),
    ])
    assert sum(isinstance(f, InterruptionFrame) for f in down) == 1
    assert [d["text"].strip() for k, d in setup.events if k == "answer"] == ["Okay."]
    assert setup.session.messages[-2] == {"role": "user", "content": "stop"}


async def test_interruption_while_speaking_cancels_turn():
    llm = FakeLlm(["It costs five hundred fifty taka for one board today."], delay_s=0.1)
    setup = Setup(llm)
    down = await setup.run([
        said("price of esp32"), SleepFrame(0.25),
        BotStartedSpeakingFrame(), SleepFrame(0.05),
        InterruptionFrame(), SleepFrame(0.1),
    ])
    assert "answer" not in setup.kinds()
    assert any(isinstance(f, InterruptionFrame) for f in down)
    assert not any(isinstance(f, LLMFullResponseEndFrame) for f in down)
    assert setup.session.messages == [{"role": "user", "content": "price of esp32"}]


async def test_bot_audio_of_filler_does_not_arm_the_cancel():
    llm = FakeLlm([[("search_products", '{"text":"esp32"}')], "It costs 550 taka."],
                  delay_s=0.1)
    setup = Setup(llm)
    down = await setup.run([
        said("price of esp32"), SleepFrame(0.15),
        BotStartedSpeakingFrame(), SleepFrame(0.05),
        InterruptionFrame(), SleepFrame(0.05),
    ])
    assert any(isinstance(f, TTSSpeakFrame) for f in down)
    assert not any(isinstance(f, InterruptionFrame) for f in down)
    assert setup.kinds()[-1] == "answer"
    assert setup.session.messages[-1] == {"role": "assistant", "content": "It costs 550 taka."}


def summary_of(session):
    (summary,) = session.summaries.values()
    return summary


async def test_answer_audio_that_did_not_end_is_not_heard():
    setup = Setup(FakeLlm([[("prepare_order", PREPARE)], SUMMARY_TEXT]))
    await setup.run([
        said("order one esp32"), SleepFrame(0.3),
        *audio_pair(),  # the filler audio
        BotStartedSpeakingFrame(), SleepFrame(0.05),
    ])
    assert summary_of(setup.session).heard is False
    assert "heard" not in setup.kinds()


async def test_heard_after_filler_audio_and_full_answer():
    setup = Setup(FakeLlm([[("prepare_order", PREPARE)], SUMMARY_TEXT]))
    await setup.run([
        said("order one esp32"), SleepFrame(0.3),
        *audio_pair(),  # the filler audio
        *audio_pair(),  # the answer audio
    ])
    assert summary_of(setup.session).heard is True
    assert ("heard", 1) in [(k, d.get("turn")) for k, d in setup.events]


async def test_fallback_answer_leaves_no_heard_summary():
    # Round 2 is empty, so the caller hears only the fallback line, not the summary.
    setup = Setup(FakeLlm([[("prepare_order", PREPARE)], ""]))
    await setup.run([
        said("order one esp32"), SleepFrame(0.3),
        *audio_pair(),  # the filler audio
        *audio_pair(),  # the fallback audio
    ])
    assert setup.session.summaries == {}
    assert not any(k == "heard" and d.get("summary_ids") for k, d in setup.events)


async def test_interrupted_summary_is_not_heard(monkeypatch):
    monkeypatch.setattr("support_agent.tools.uuid.uuid4", lambda: uuid.UUID(int=0))
    place = json.dumps({"summary_id": "00000000"})
    store = FakeStore()
    llm = FakeLlm([[("prepare_order", PREPARE)], SUMMARY_TEXT,
                   # Turn 2: place_order fails, so the model makes a new summary.
                   [("place_order", place)], [("prepare_order", PREPARE)], SUMMARY_TEXT,
                   # Turn 3: the caller heard the new summary in full.
                   [("place_order", place)], "Your order is placed."])
    setup = Setup(llm, store)
    await setup.run([
        said("order one esp32"), SleepFrame(0.3),
        *audio_pair(),  # the filler audio
        BotStartedSpeakingFrame(), SleepFrame(0.05),
        InterruptionFrame(), SleepFrame(0.05),
        BotStoppedSpeakingFrame(), SleepFrame(0.05),
        said("yes"), SleepFrame(0.3),
        *audio_pair(),  # the filler audio of turn 2
        *audio_pair(),  # the new summary of turn 2, in full
        said("yes"), SleepFrame(0.3),
    ])
    # The interrupted summary of turn 1 is not heard. The new summary of turn 2 is heard.
    assert [d.get("turn") for k, d in setup.events if k == "heard"] == [2]
    place_end = [d for k, d in setup.events if k == "tool_end" and d["name"] == "place_order"]
    assert [d["ok"] for d in place_end] == [False, True]
    assert "Call prepare_order again" in place_end[0]["content"]
    assert len(store.placed) == 1


async def test_yes_during_the_summary_audio_does_not_place_the_order(monkeypatch):
    monkeypatch.setattr("support_agent.tools.uuid.uuid4", lambda: uuid.UUID(int=0))
    place = json.dumps({"summary_id": "00000000"})
    store = FakeStore()
    llm = FakeLlm([[("prepare_order", PREPARE)], SUMMARY_TEXT,
                   # Turn 2 is the "okay" that the caller said during the summary audio.
                   [("place_order", place)], "Please listen to the full summary first."])
    setup = Setup(llm, store)
    await setup.run([
        said("order one esp32"), SleepFrame(0.3),
        *audio_pair(),  # the filler audio of turn 1
        BotStartedSpeakingFrame(), SleepFrame(0.05),  # the summary audio of turn 1
        said("okay"), SleepFrame(0.3),
    ])
    assert "heard" not in setup.kinds()
    place_end = [d for k, d in setup.events if k == "tool_end" and d["name"] == "place_order"]
    assert [d["ok"] for d in place_end] == [False]
    assert store.placed == []


async def test_history_keeps_only_the_spoken_part_of_an_interrupted_answer():
    llm = MixedLlm([("Sure.", [("search_products", '{"text":"esp32"}')]),
                    "It costs 550 taka. It is in stock."])
    setup = Setup(llm)
    await setup.run_with_tracker([
        said("price of esp32"), SleepFrame(0.3),
        BotStartedSpeakingFrame(), SleepFrame(0.05),
        spoken("Sure."), spoken("It costs 550 taka."), SleepFrame(0.05),
        InterruptionFrame(), SleepFrame(0.05),
        spoken("It is in stock."), SleepFrame(0.05),
    ])
    assert setup.session.messages[-1] == {"role": "assistant", "content": "It costs 550 taka."}
    assert setup.session.messages[1]["content"] == "Sure."


async def test_unheard_answer_leaves_a_valid_history():
    llm = FakeLlm([[("search_products", '{"text":"esp32"}')], "It costs 550 taka."])
    setup = Setup(llm)
    await setup.run_with_tracker([
        said("price of esp32"), SleepFrame(0.3),
        spoken(FILLER_PHRASES[0], append_to_context=False),
        *audio_pair(),  # the filler audio
        BotStartedSpeakingFrame(), SleepFrame(0.05),
        InterruptionFrame(), SleepFrame(0.05),
    ])
    roles = [m["role"] for m in setup.session.messages]
    assert roles == ["user", "assistant", "tool"]
    call_ids = [c["id"] for c in setup.session.messages[1]["tool_calls"]]
    assert call_ids == [setup.session.messages[2]["tool_call_id"]]


async def test_history_is_not_changed_without_a_tracker():
    setup = Setup(FakeLlm(["It costs 550 taka. It is in stock."]))
    await setup.run([
        said("price of esp32"), SleepFrame(0.3),
        BotStartedSpeakingFrame(), SleepFrame(0.05),
        InterruptionFrame(), SleepFrame(0.05),
    ])
    assert setup.session.messages[-1]["content"] == "It costs 550 taka. It is in stock."


async def test_words_replace_an_answer_whose_audio_did_not_start():
    llm = FakeLlm(["It costs 550 taka.", "Yes, I am here."])
    setup = Setup(llm)
    setup.processor._answer_audio_timeout_s = 5.0
    started: list[float] = []

    async def on_event(kind, data):
        setup.events.append((kind, data))
        if kind == "turn_start":
            started.append(asyncio.get_running_loop().time())

    setup.processor._on_event = on_event
    await setup.run([said("price of esp32"), SleepFrame(0.05), said("hello?"), SleepFrame(1.5)])
    # The second turn does not wait for the audio of the first answer.
    assert len(started) == 2 and started[1] - started[0] < 0.5
    assert [d["text"].strip() for k, d in setup.events if k == "answer"][-1] == "Yes, I am here."


async def test_greeting_is_spoken_and_kept_in_the_history():
    from support_agent.voice.agent_processor import GreetingFrame

    setup = Setup(FakeLlm(["Sure, the ESP32 costs five hundred fifty taka."]))
    down = await setup.run([
        GreetingFrame("Hi, thanks for calling Robo Parts BD."), SleepFrame(0.05),
        *audio_pair(),  # the greeting audio
        said("price of esp32"), SleepFrame(0.3),
    ])
    speak = [f for f in down if isinstance(f, TTSSpeakFrame)]
    assert [f.text for f in speak] == ["Hi, thanks for calling Robo Parts BD."]
    assert setup.session.messages[0] == {"role": "assistant",
                                         "content": "Hi, thanks for calling Robo Parts BD."}
    # The model sees the greeting, so it does not greet again.
    assert setup.llm.seen[0][1]["content"] == "Hi, thanks for calling Robo Parts BD."
    assert ("greeting", "Hi, thanks for calling Robo Parts BD.") in [
        (k, d.get("text")) for k, d in setup.events]


async def test_speech_during_the_greeting_does_not_stop_it():
    from support_agent.voice.agent_processor import GreetingFrame

    setup = Setup(FakeLlm(["Hello!"]))
    down = await setup.run([
        GreetingFrame("Hi there."), SleepFrame(0.02),
        BotStartedSpeakingFrame(), SleepFrame(0.02),
        InterruptionFrame(), SleepFrame(0.05),
    ])
    assert not any(isinstance(f, InterruptionFrame) for f in down)
