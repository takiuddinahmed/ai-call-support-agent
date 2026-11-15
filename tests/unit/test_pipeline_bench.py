import asyncio
import json
import time

from pipecat.frames.frames import InputAudioRawFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from support_agent.loop import AgentEvent
from support_agent.voice.bench.pipeline_bench import (
    FileAudioTransport,
    ProcessMemoryPoller,
    TurnRecord,
    answer_audio_done,
    answer_with_digits,
    check_summary_city,
    fill_caller_text,
    keep_last_answer,
    memory_upper_gb,
    parse_ollama_processes,
    place_setup_order,
    placed_order_number,
    rebuild_report,
    render_report,
    slept_seconds,
    turn_metrics,
    words_to_digits,
    write_report,
)

RATE = 16000


def caller_turn(value: int, seconds: float = 0.2) -> bytes:
    """Caller audio of one turn: every byte has `value`, so a fake can tell the turns apart."""
    return bytes([value]) * (int(RATE * seconds) * 2)


class FakeAgent(FrameProcessor):
    """Answers each caller turn with `pairs` agent audio parts. Each part ends with a stop.

    The first part of a turn with two parts acts as the filler.
    """

    def __init__(self, pairs: int = 1, answer: bool = True, silent_end=None):
        super().__init__()
        self.pairs = pairs
        self.answer = answer
        # A transport. If it is set, the fake gives no audio and ends the turn silently.
        self.silent_end = silent_end
        self.heard: list[int] = []
        self._in_speech = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            loud = any(frame.audio)
            if loud and not self._in_speech:
                self._in_speech = True
                self.heard.append(frame.audio[0])
            elif not loud and self._in_speech:
                self._in_speech = False
                if self.silent_end is not None:
                    self.silent_end.end_turn_when_silent()
                elif self.answer:
                    self.create_task(self._speak())
            return
        await self.push_frame(frame, direction)

    async def _speak(self):
        for _ in range(self.pairs):
            await self.push_frame(TTSAudioRawFrame(b"\x01\x00" * 1600, RATE, 1))
            await self.push_frame(TTSStoppedFrame())
            await asyncio.sleep(0.3)


async def run_call(transport: FileAudioTransport, agent: FakeAgent, limit_s: float = 15.0):
    worker = PipelineWorker(
        Pipeline([transport.input(), agent, transport.output()]),
        params=PipelineParams(audio_in_sample_rate=RATE, audio_out_sample_rate=RATE),
        enable_rtvi=False,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await asyncio.wait_for(runner.run(), limit_s)


async def test_file_transport_plays_turns_in_order():
    transport = FileAudioTransport([caller_turn(1), caller_turn(2)], RATE,
                                   lead_s=0.1, silence_s=0.1, turn_timeout_s=5)
    agent = FakeAgent()
    await run_call(transport, agent)
    assert agent.heard == [1, 2]
    assert len(transport.records) == 2
    assert all(record.agent_audio for record in transport.records)
    assert not any(record.timed_out for record in transport.records)
    # The caller audio of turn 2 starts after the agent audio of turn 1 ends.
    assert transport.records[1].caller_start > transport.records[0].bot_stopped[-1]


async def test_file_transport_paces_the_caller_audio_in_real_time():
    transport = FileAudioTransport([caller_turn(1, seconds=0.6)], RATE,
                                   lead_s=0.1, silence_s=0.1, turn_timeout_s=5)
    await run_call(transport, FakeAgent())
    record = transport.records[0]
    assert record.caller_end - record.caller_start >= 0.55


async def test_turn_waits_for_the_answer_audio_and_not_the_filler():
    records: list[TurnRecord] = []

    def answer_done(record: TurnRecord) -> bool:
        # The second stop of a turn ends the answer audio. The first stop ends the filler.
        return len(record.bot_stopped) >= 2

    transport = FileAudioTransport([caller_turn(1), caller_turn(2)], RATE, lead_s=0.1,
                                   silence_s=0.1, turn_timeout_s=5, turn_done=answer_done)
    agent = FakeAgent(pairs=2)
    await run_call(transport, agent)
    records = transport.records
    assert agent.heard == [1, 2]
    assert len(records[0].bot_started) == 2
    assert records[1].caller_start > records[0].bot_stopped[1]


async def test_stuck_turn_times_out_and_ends_the_call():
    transport = FileAudioTransport([caller_turn(1), caller_turn(2)], RATE,
                                   lead_s=0.1, silence_s=0.1, turn_timeout_s=0.5)
    agent = FakeAgent(answer=False)
    started = time.monotonic()
    await run_call(transport, agent)
    assert time.monotonic() - started < 5
    assert agent.heard == [1]
    assert transport.records[0].timed_out
    assert len(transport.records) == 1


def test_words_to_digits():
    assert words_to_digits("It costs five hundred fifty taka.") == "It costs 550 taka."
    assert words_to_digits("one thousand two hundred and fifty taka") == "1250 taka"
    assert words_to_digits("twelve hundred fifty") == "1250"
    assert words_to_digits("Order six is delivered") == "Order 6 is delivered"
    assert words_to_digits("one hundred twenty-five") == "125"
    assert words_to_digits("and then") == "and then"


def test_answer_with_digits_changes_only_the_answer_event():
    events = [AgentEvent("tool_start", 1.0, {"name": "get_product"}),
              AgentEvent("answer", 2.0, {"text": "It is five hundred fifty taka."})]
    out = answer_with_digits(events)
    assert out[0] is events[0]
    assert out[1].data["text"] == "It is 550 taka."


def test_turn_metrics_measures_from_the_caller_audio_end():
    record = TurnRecord(index=0, caller_start=10.0, caller_end=12.0,
                        bot_started=[12.8, 15.0], bot_stopped=[13.5, 18.0])
    filler = turn_metrics(record, had_filler=True)
    assert filler == {"first_audio_s": 0.8, "filler_s": 0.8, "answer_s": 3.0,
                      "talked_over": False}
    plain = turn_metrics(TurnRecord(index=0, caller_start=0.0, caller_end=1.0,
                                    bot_started=[2.5], bot_stopped=[4.0]), had_filler=False)
    assert plain == {"first_audio_s": 1.5, "filler_s": None, "answer_s": 1.5,
                     "talked_over": False}
    silent = turn_metrics(TurnRecord(index=0, caller_start=0.0, caller_end=1.0),
                          had_filler=False)
    assert silent == {"first_audio_s": None, "filler_s": None, "answer_s": None,
                      "talked_over": False}


def test_turn_metrics_ignores_agent_audio_that_started_during_the_caller_audio():
    # The turn split: the agent answered the first sentence while the caller still spoke.
    record = TurnRecord(index=0, caller_start=0.0, caller_end=5.0,
                        bot_started=[3.0, 6.0], bot_stopped=[4.0, 8.0])
    metrics = turn_metrics(record, had_filler=False)
    assert metrics == {"first_audio_s": 1.0, "filler_s": None, "answer_s": 1.0,
                       "talked_over": True}


def test_answer_audio_done_waits_for_the_text_after_the_caller_audio():
    record = TurnRecord(index=0, caller_start=0.0, caller_end=5.0)
    # One transcription during the caller audio, and its answer. The caller still speaks.
    assert not answer_audio_done(record, text_times=[3.0], answers=1)
    # The last transcription came after the caller audio. Its answer did not come yet.
    assert not answer_audio_done(record, text_times=[3.0, 5.6], answers=1)
    assert answer_audio_done(record, text_times=[3.0, 5.6], answers=2)
    assert answer_audio_done(record, text_times=[5.6], answers=1)
    assert not answer_audio_done(record, text_times=[], answers=0)


def test_keep_last_answer_drops_the_answers_of_earlier_agent_turns():
    events = [AgentEvent("answer", 1.0, {"text": "first"}),
              AgentEvent("tool_start", 2.0, {"name": "prepare_order"}),
              AgentEvent("answer", 3.0, {"text": "second"})]
    out = keep_last_answer(events)
    assert [e.data.get("text") for e in out if e.type == "answer"] == ["second"]
    assert any(e.type == "tool_start" for e in out)


def test_parse_ollama_processes_sums_rss_and_finds_the_runner():
    ps = (" 319 3845248 /Applications/Ollama.app/Contents/Resources/llama-server --model /x\n"
          " 979   16688 /Applications/Ollama.app/Contents/MacOS/Ollama hidden\n"
          "1005   27632 /Applications/Ollama.app/Contents/Resources/ollama serve\n"
          "2000   50000 /usr/bin/python something else\n")
    rss, runners = parse_ollama_processes(ps)
    assert rss == (3845248 + 16688 + 27632) * 1024
    assert runners == [319]


def _turn(call, tools, first, answer, filler=None):
    return {"call": call, "turn": 1, "heard": "hi", "tools": tools, "first_audio_s": first,
            "filler_s": filler, "answer_s": answer, "timer_pause_ms": None, "agent_turns": 1,
            "answer": "ok", "failures": []}


def _result(memory_upper_parts=(4.0, 4.9, 3.1), passed=True):
    agent, runner, estimate = memory_upper_parts
    return {
        "rate": 8000,
        "date": "2026-11-15",
        "model": "m",
        "stt_model": "s",
        "run_id": "abc",
        "seconds": 10.0,
        "calls": [{"id": "c1", "passed": passed, "failures": [] if passed else ["turn 1: x"],
                   "turns": [_turn("c1", [], 1.0, 1.0),
                             _turn("c1", ["get_product"], 0.5, 3.0, filler=0.5),
                             {**_turn("c1", [], 9.0, 9.0), "split": True}]},
                  {"id": "long", "passed": True, "failures": [], "turns": [],
                   "memory": {"agent_footprint_gb": 3.5, "runner_footprint_gb": 4.0,
                              "ollama_rss_gb": 3.9, "estimate_gb": 3.1}}],
        "memory": {"agent_peak_rss_gb": 1.5, "agent_footprint_peak_gb": agent,
                   "ollama_peak_rss_gb": 0.4, "ollama_runner_footprint_gb": runner,
                   "ollama_estimate_gb": estimate},
    }


def test_report_has_the_parts():
    text = render_report(_result())
    assert "G.711 fallback" in text
    for heading in ("## Goals", "## Pause", "## Filler", "## Calls", "## Memory",
                    "## Known findings"):
        assert heading in text
    assert "| c1 | pass |" in text
    assert "single run, rough baseline" in text
    assert "Sample: 2 calls, 3 turns." in text
    # The split turn is in the numbers, and it also has its own row.
    assert "| Without tools | 2 | 5.00 |" in text
    assert "| Split turns | 1 | 9.00 |" in text
    assert "| All turns | 3 |" in text
    assert "-4.27 s" in text
    assert "mapped" in text
    # The memory of the long call.
    assert "| long | 0 | 3.5 | 4.0 | 3.9 | 3.1 | 10.6 |" in text


def test_report_goals_give_met_or_missed():
    text = render_report(_result())
    assert text.count("goal, not a gate") == 1
    assert "| Pause median, turns without tools | 1.5 s or less | 5.00 s | missed |" in text
    assert "| Filler start median, turns with tools | 1.5 s or less | 0.50 s | met |" in text
    assert "| Memory upper figure | 12 GB or less | 12.0 GB | met |" in text
    assert "| Calls passed | all | 2 of 2 | met |" in text
    missed = render_report(_result(memory_upper_parts=(5.0, 5.0, 3.0), passed=False))
    assert "| Memory upper figure | 12 GB or less | 13.0 GB | missed |" in missed
    assert "| Calls passed | all | 1 of 2 | missed |" in missed


def test_memory_upper_figure_adds_the_three_parts():
    assert memory_upper_gb({"agent_footprint_peak_gb": 4.0, "ollama_runner_footprint_gb": 4.9,
                            "ollama_estimate_gb": 3.1}) == 12.0


def test_memory_poller_keeps_the_run_peak_and_the_call_peak():
    poller = ProcessMemoryPoller(api_ps_url="")
    poller.record({"agent_footprint": 3e9, "runner_footprint": 4e9, "ollama_rss": 1e9,
                   "estimate": 3e9})
    poller.start_window()
    poller.record({"agent_footprint": 2e9, "runner_footprint": 5e9, "ollama_rss": 1e9,
                   "estimate": 3e9})
    assert poller.window == {"agent_footprint": 2e9, "runner_footprint": 5e9,
                             "ollama_rss": 1e9, "estimate": 3e9}
    assert poller.peak["agent_footprint"] == 3e9
    assert poller.peak["runner_footprint"] == 5e9


def test_write_report_keeps_the_raw_result_and_rebuilds_the_report(tmp_path):
    path = write_report(_result(), tmp_path)
    raw = path.with_suffix(".json")
    assert json.loads(raw.read_text())["run_id"] == "abc"
    text = path.read_text()
    path.unlink()
    assert rebuild_report(raw) == path
    assert path.read_text() == text


def test_summary_city_check_reads_the_order_summary_address():
    def summary(city):
        content = json.dumps({"summary_id": "a", "address": {"city": city}})
        return [AgentEvent("tool_end", 1.0, {"name": "prepare_order", "ok": True,
                                             "content": content})]

    assert check_summary_city({"summary_city": "Sylhet"}, summary(" sylhet ")) == []
    assert check_summary_city({"summary_city": "Sylhet"}, summary("Sillet")) == [
        "order summary city: Sillet, the caller said: Sylhet"]
    assert check_summary_city({}, summary("Sillet")) == []
    assert check_summary_city({"summary_city": "Sylhet"}, []) == [
        "no order summary to check the city"]


def test_placed_order_number_reads_the_place_order_result():
    events = [AgentEvent("tool_end", 1.0, {"name": "prepare_order", "ok": True,
                                           "content": '{"summary_id":"abc"}'}),
              AgentEvent("tool_end", 2.0, {"name": "place_order", "ok": True,
                                           "content": '{"order":{"display_id":44}}'})]
    assert placed_order_number(events) == 44
    failed = [AgentEvent("tool_end", 1.0, {"name": "place_order", "ok": False,
                                           "content": "Unknown summary_id."})]
    assert placed_order_number(failed) is None


def test_fill_caller_text_puts_in_the_bench_order_number():
    assert fill_caller_text("The order number is {bench_order}.", 44) == \
        "The order number is 44."
    assert fill_caller_text("Hello.", None) == "Hello."
    assert fill_caller_text("Order {bench_order}.", None) is None


def test_calls_file_uses_known_checks_and_has_a_20_turn_call():
    from support_agent.bench.runner import load_cases
    from support_agent.voice.bench.pipeline_bench import CALLS_PATH

    known = {"tools_called", "tools_not_called", "answer_contains_any", "answer_contains_all",
             "answer_not_contains", "order_placed", "no_list_lines", "summary_city"}
    cases = load_cases(str(CALLS_PATH))
    for case in cases:
        for turn in case.turns:
            assert set(turn.get("expect", {})) <= known, (case.id, turn["user"])
    memory_call = next(c for c in cases if c.id == "voice-long-call-memory")
    assert len(memory_call.turns) == 20
    assert not any("place the order" in t["user"] for t in memory_call.turns)


async def test_turn_with_no_agent_audio_ends_when_the_agent_is_silent():
    # An empty answer gives no audio and no stop frame. The bench ends the turn by itself.
    transport = FileAudioTransport([caller_turn(1), caller_turn(2)], RATE, lead_s=0.1,
                                   silence_s=0.1, turn_timeout_s=5,
                                   turn_done=lambda record: False)
    agent = FakeAgent(silent_end=transport)
    started = time.monotonic()
    await run_call(transport, agent)
    assert agent.heard == [1, 2]
    assert not any(record.timed_out for record in transport.records)
    assert time.monotonic() - started < 4


async def test_place_setup_order_uses_the_bench_email_and_phone():
    from support_agent.store.models import Product, Variant
    from tests.unit.fakes import FakeStore

    store = FakeStore()
    breadboard = Product(id="p-1", handle="breadboard-830", title="Breadboard", description="",
                         category=None, tags=[], specs={},
                         variants=[Variant("v-1", "Default", None, 120, 5)])

    async def get_product(handle):
        return breadboard if handle == "breadboard-830" else None

    store.get_product = get_product
    number = await place_setup_order(store, "bench-x@example.com")
    assert number == 10
    items, address, email, token = store.placed[0]
    assert items == [("v-1", 1)]
    assert email == "bench-x@example.com"
    assert address.phone == "01811111111"
    assert token is None


def test_report_states_empty_answers_and_the_context_length():
    result = _result(passed=False)
    result["context_length"] = 4096
    result["setup_order"] = 50
    result["calls"][0]["failures"] = ["turn 1: the agent gave an empty answer"]
    text = render_report(result)
    assert "Ollama context: 4096 tokens." in text
    assert "Setup order: 50." in text
    assert "Empty answers: 1." in text


def test_slept_seconds_is_the_wall_time_that_the_monotonic_clock_did_not_count():
    # macOS stops time.monotonic() during sleep. time.time() goes on.
    assert slept_seconds(wall_s=1000.0, monotonic_s=400.0) == 600.0
    assert slept_seconds(wall_s=400.5, monotonic_s=400.0) == 0.0


def test_report_marks_a_run_with_sleep_as_not_valid():
    result = _result()
    result["slept_s"] = 600.0
    text = render_report(result)
    assert "NOT VALID: the computer slept for 600 s during the run." in text
    result["slept_s"] = 0.0
    assert "NOT VALID" not in render_report(result)
