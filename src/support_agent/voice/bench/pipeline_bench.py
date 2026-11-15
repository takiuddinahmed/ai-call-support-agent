"""Full pipeline benchmark: scripted caller turns go as audio through the real voice pipeline.

FileAudioTransport is a Pipecat transport that acts as a caller. It plays each caller turn
as audio in real time, then sends silence. It waits until the agent audio of that turn ends,
then it plays the next turn. It records the agent audio and the times of each turn.

The benchmark runs the calls in calls.yaml. The calls use the same shape and the same checks
as the chat benchmark cases (support_agent.bench.runner). The report has the pause of each
turn, the filler start time, the pass or fail of each call, and the peak memory.
"""

import asyncio
import contextlib
import json
import os
import re
import resource
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelWorkerFrame,
    EndWorkerFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from support_agent.loop import EMPTY_ANSWER_FALLBACK, AgentEvent
from support_agent.voice.bench.stt import percentile
from support_agent.voice.debug_hub import DebugHub

CALLS_PATH = Path(__file__).with_name("calls.yaml")
REPORTS_DIR = Path("reports")
# The caller voice. It must not be the agent voice (settings.voice_tts_voice).
CALLER_VOICES = ("am_adam", "bm_george")
# A stuck turn fails the call after this time and does not hang the run.
TURN_TIMEOUT_S = 60.0
# Silence before the first turn, and after the agent audio of a turn ends.
LEAD_S = 1.0
SILENCE_S = 1.0
FRAME_MS = 20
SAMPLE_BYTES = 2
# In the caller text of calls.yaml, the number of the order that this run placed.
BENCH_ORDER = "{bench_order}"
# The event types of run_turn(). The checks of the chat benchmark read these events.
AGENT_EVENT_TYPES = {
    "turn_start", "llm_start", "first_token", "text", "tool_call_start", "llm_end",
    "tool_start", "tool_end", "answer", "error",
}


# The transport


@dataclass
class TurnRecord:
    """The times of one caller turn, in time.monotonic() seconds, and the agent audio."""

    index: int
    caller_start: float = 0.0
    caller_end: float = 0.0
    bot_started: list[float] = field(default_factory=list)
    bot_stopped: list[float] = field(default_factory=list)
    agent_audio: bytearray = field(default_factory=bytearray)
    timed_out: bool = False


TurnDone = Callable[[TurnRecord], bool]


class FileAudioTransport(BaseTransport):
    """A Pipecat transport that plays caller turns from memory, like a real caller.

    Each turn is 16-bit mono PCM at `sample_rate`. The input sends one frame each 20 ms in
    real time: silence first, then the turn audio, then silence until the agent audio of the
    turn ends. Then it sends `silence_s` of silence and plays the next turn.

    The agent audio of a turn ends at a BotStoppedSpeakingFrame for which `turn_done` gives
    True. If `turn_done` is None, the first BotStoppedSpeakingFrame ends the turn. The
    benchmark gives a `turn_done` that ignores the stop after the filler audio.

    If a turn does not end in `turn_timeout_s`, the transport marks it as timed out and
    cancels the call. After the last turn, it ends the call.

    The output paces the agent audio in real time and records it in the turn records.
    """

    def __init__(self, turns: list[bytes], sample_rate: int, *, lead_s: float = LEAD_S,
                 silence_s: float = SILENCE_S, turn_timeout_s: float = TURN_TIMEOUT_S,
                 turn_done: TurnDone | None = None, **kwargs):
        super().__init__(**kwargs)
        self.turns = turns
        self.sample_rate = sample_rate
        self.lead_s = lead_s
        self.silence_s = silence_s
        self.turn_timeout_s = turn_timeout_s
        self._turn_done = turn_done
        self.records: list[TurnRecord] = []
        self._turn_ended = asyncio.Event()
        self._bot_speaking = False
        self._end_when_silent = False
        self._params = TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=sample_rate,
            audio_out_sample_rate=sample_rate,
        )
        self._input: _FileInput | None = None
        self._output: _FileOutput | None = None

    def input(self) -> "_FileInput":
        if self._input is None:
            self._input = _FileInput(self, self._params, name=self._input_name)
        return self._input

    def output(self) -> "_FileOutput":
        if self._output is None:
            self._output = _FileOutput(self, self._params, name=self._output_name)
        return self._output

    @property
    def current(self) -> TurnRecord | None:
        return self.records[-1] if self.records else None

    def begin_turn(self) -> TurnRecord:
        record = TurnRecord(index=len(self.records))
        self.records.append(record)
        self._turn_ended.clear()
        self._end_when_silent = False
        return record

    def end_turn_when_silent(self) -> None:
        """Ends the turn now if no agent audio plays, or at the next stop if audio plays.

        Use it when the agent will give no more audio in this turn, for example after an
        empty answer or an error. Such a turn gives no stop frame after the answer.
        """
        self._end_when_silent = True
        if not self._bot_speaking:
            self._turn_ended.set()

    def on_bot_started(self) -> None:
        self._bot_speaking = True
        if self.current is not None:
            self.current.bot_started.append(time.monotonic())

    def on_bot_stopped(self) -> None:
        self._bot_speaking = False
        record = self.current
        if record is None:
            return
        record.bot_stopped.append(time.monotonic())
        if self._end_when_silent or self._turn_done is None or self._turn_done(record):
            self._turn_ended.set()

    def on_agent_audio(self, audio: bytes) -> None:
        if self.current is not None:
            self.current.agent_audio.extend(audio)

    async def wait_turn_end(self, timeout_s: float) -> bool:
        try:
            await asyncio.wait_for(self._turn_ended.wait(), timeout_s)
        except TimeoutError:
            return False
        return True


class _Pacer:
    """Sleeps so that each call of `tick` comes one period after the previous one."""

    def __init__(self, period_s: float):
        self.period_s = period_s
        self._next = 0.0

    def reset(self) -> None:
        self._next = 0.0

    async def tick(self) -> None:
        now = time.monotonic()
        if self._next <= now:
            self._next = now + self.period_s
            return
        await asyncio.sleep(self._next - now)
        self._next += self.period_s


class _FileInput(BaseInputTransport):
    def __init__(self, transport: FileAudioTransport, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._feed_task: asyncio.Task | None = None
        self._frame_bytes = transport.sample_rate * FRAME_MS // 1000 * SAMPLE_BYTES
        self._silence = bytes(self._frame_bytes)
        self._pacer = _Pacer(FRAME_MS / 1000)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        if self._feed_task is None:
            self._feed_task = self.create_task(self._feed(), name="file_audio_feed")

    async def stop(self, frame):
        await self._stop_feed()
        await super().stop(frame)

    async def cancel(self, frame):
        await self._stop_feed()
        await super().cancel(frame)

    async def cleanup(self):
        await self._stop_feed()
        await super().cleanup()

    async def _stop_feed(self) -> None:
        task = self._feed_task
        self._feed_task = None
        if task is not None and task is not asyncio.current_task():
            await self.cancel_task(task)

    async def _send(self, audio: bytes) -> None:
        if len(audio) < self._frame_bytes:
            audio = audio + bytes(self._frame_bytes - len(audio))
        await self._pacer.tick()
        await self.push_audio_frame(InputAudioRawFrame(
            audio=audio, sample_rate=self._transport.sample_rate, num_channels=1))

    async def _send_silence(self, seconds: float) -> None:
        for _ in range(round(seconds * 1000 / FRAME_MS)):
            await self._send(self._silence)

    async def _silence_until_turn_end(self) -> bool:
        """Sends silence until the agent audio of the turn ends. False after the timeout."""
        transport = self._transport
        waiter = asyncio.ensure_future(transport.wait_turn_end(transport.turn_timeout_s))
        try:
            while not waiter.done():
                await self._send(self._silence)
            return waiter.result()
        finally:
            waiter.cancel()

    async def _feed(self) -> None:
        transport = self._transport
        await self._send_silence(transport.lead_s)
        for audio in transport.turns:
            record = transport.begin_turn()
            record.caller_start = time.monotonic()
            for start in range(0, len(audio), self._frame_bytes):
                await self._send(audio[start:start + self._frame_bytes])
            record.caller_end = time.monotonic()
            if not await self._silence_until_turn_end():
                record.timed_out = True
                logger.warning(f"{self}: turn {record.index + 1} did not end in "
                               f"{transport.turn_timeout_s} s. The call stops.")
                await self.push_frame(CancelWorkerFrame(reason="A turn timed out."),
                                      FrameDirection.UPSTREAM)
                return
            await self._send_silence(transport.silence_s)
        await self.push_frame(EndWorkerFrame(reason="The caller ended the call."),
                              FrameDirection.UPSTREAM)


class _FileOutput(BaseOutputTransport):
    def __init__(self, transport: FileAudioTransport, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._pacer = _Pacer(0.0)

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        self._pacer.period_s = self.audio_chunk_size / (self.sample_rate * SAMPLE_BYTES)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._pacer.reset()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        # The media sender pushes each bot speaking frame in both directions. The transport
        # counts the upstream copy only.
        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._transport.on_bot_started()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._transport.on_bot_stopped()
        await super().push_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        self._transport.on_agent_audio(frame.audio)
        await self._pacer.tick()
        return True


# Answer checks


UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100_000, "million": 1_000_000}
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|[^A-Za-z]+")


def _is_number_word(word: str) -> bool:
    return all(part in UNITS or part in SCALES for part in word.lower().split("-"))


def _value(words: list[str]) -> int:
    total = current = 0
    for word in words:
        for part in word.lower().split("-"):
            if part in UNITS:
                current += UNITS[part]
            elif part == "hundred":
                current = max(current, 1) * 100
            else:
                total += max(current, 1) * SCALES[part]
                current = 0
    return total + current


def words_to_digits(text: str) -> str:
    """Changes English number words to digits: "five hundred fifty" becomes "550".

    The voice agent says numbers in words. The chat benchmark checks look for digits, so the
    benchmark changes the answer text before the checks. "and" between number words is part
    of the number. The function does not change other text.
    """
    tokens = _WORD.findall(text)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if not _is_number_word(tokens[i]):
            out.append(tokens[i])
            i += 1
            continue
        words = [tokens[i]]
        j = i + 1
        # A number continues over one space, and over " and " before another number word.
        while j + 1 < len(tokens) and tokens[j] == " ":
            nxt = tokens[j + 1]
            if _is_number_word(nxt):
                words.append(nxt)
                j += 2
            elif (nxt.lower() == "and" and j + 3 < len(tokens) and tokens[j + 2] == " "
                  and _is_number_word(tokens[j + 3])):
                words.append(tokens[j + 3])
                j += 4
            else:
                break
        out.append(str(_value(words)))
        i = j
    return "".join(out)


def answer_with_digits(events: list[AgentEvent]) -> list[AgentEvent]:
    """Returns the events with the number words of the "answer" text changed to digits."""
    return [replace(e, data={**e.data, "text": words_to_digits(e.data.get("text", ""))})
            if e.type == "answer" else e for e in events]


def keep_last_answer(events: list[AgentEvent]) -> list[AgentEvent]:
    """Removes all "answer" events but the last one.

    If the pipeline splits one caller turn into two agent turns, the last answer is the
    answer to the full caller turn. The checks read the first answer event, so the earlier
    answers must go.
    """
    answers = [i for i, e in enumerate(events) if e.type == "answer"]
    drop = set(answers[:-1])
    return [e for i, e in enumerate(events) if i not in drop]


def answer_audio_done(record: TurnRecord, text_times: list[float], answers: int) -> bool:
    """True if the agent answered all caller text of the turn, also the last part.

    `text_times` are the times of the "user_text" events of the turn. The last part of the
    caller audio gives a "user_text" event after the end of the caller audio. If the
    pipeline split the turn, an earlier part gives one during the caller audio. Each
    "user_text" event starts one agent turn, and each agent turn gives one answer or error.
    """
    if not any(t >= record.caller_end for t in text_times):
        return False
    return answers >= len(text_times)


def placed_order_number(events: list[AgentEvent]) -> int | None:
    """The order number of the last order that place_order placed in the events, or None."""
    number = None
    for e in events:
        if e.type == "tool_end" and e.data.get("name") == "place_order" and e.data.get("ok"):
            try:
                number = int(json.loads(e.data["content"])["order"]["display_id"])
            except (KeyError, TypeError, ValueError):
                continue
    return number


def fill_caller_text(text: str, bench_order: int | None) -> str | None:
    """Puts the number of the order that this run placed in place of "{bench_order}".

    Returns None if the text needs the order number and the run placed no order.
    """
    if BENCH_ORDER not in text:
        return text
    if bench_order is None:
        return None
    return text.replace(BENCH_ORDER, str(bench_order))


def turn_metrics(record: TurnRecord, had_filler: bool) -> dict:
    """Times of one turn in seconds, from the end of the caller audio.

    - first_audio_s: to the first agent audio. This is the pause that the caller hears.
    - filler_s: to the filler audio, if the turn had a filler.
    - answer_s: to the answer audio. With a filler, the answer audio is the second part.
    - talked_over: True if agent audio started before the caller audio ended.

    Agent audio that started before the end of the caller audio does not count in the
    times. It comes from a split turn: the agent answered the first part of the turn.
    """
    starts = [t for t in record.bot_started if t >= record.caller_end]

    def since_end(index: int) -> float | None:
        if index >= len(starts):
            return None
        return round(starts[index] - record.caller_end, 3)

    return {
        "first_audio_s": since_end(0),
        "filler_s": since_end(0) if had_filler else None,
        "answer_s": since_end(1 if had_filler else 0),
        "talked_over": len(starts) < len(record.bot_started),
    }


# Memory


def parse_ollama_processes(ps_output: str) -> tuple[int, list[int]]:
    """Reads the output of `ps -axo pid=,rss=,args=`.

    Returns the RSS sum in bytes of the processes whose command has "ollama" (any case),
    and the IDs of the model runners among them (a command with "--model" or "runner").
    """
    rss = 0
    runners = []
    for line in ps_output.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid, kb, args = int(parts[0]), int(parts[1]), parts[2]
        if "ollama" not in args.lower():
            continue
        rss += kb * 1024
        if "--model" in args or " runner" in args:
            runners.append(pid)
    return rss, runners


def _ps_all() -> str:
    return subprocess.run(["ps", "-axo", "pid=,rss=,args="], check=False, capture_output=True,
                          text=True).stdout


def _footprint(pid: int) -> tuple[int, int]:
    """The macOS footprint of a process in bytes: (now, peak over the process life)."""
    from support_agent.bench.runner import parse_footprint

    out = subprocess.run(["footprint", "-f", "bytes", str(pid)], check=False,
                         capture_output=True, text=True).stdout
    return parse_footprint(out)


def _api_ps_bytes(url: str) -> int:
    """The Ollama /api/ps size of all loaded models: the Ollama estimate of the weights."""
    import httpx

    data = httpx.get(url, timeout=5).json()
    return sum(model.get("size", 0) for model in data.get("models", []))


# A difference below this is clock noise, not sleep.
SLEEP_TOLERANCE_S = 5.0


def slept_seconds(wall_s: float, monotonic_s: float) -> float:
    """The time in seconds that the computer slept during a run.

    On macOS, time.monotonic() stops during sleep and time.time() does not. If the computer
    sleeps, the pacing and the turn times are not valid.
    """
    gap = wall_s - monotonic_s
    return round(gap, 1) if gap > SLEEP_TOLERANCE_S else 0.0


def _context_length(url: str, model: str) -> int | None:
    """The context length of the loaded model from /api/ps, or None."""
    with contextlib.suppress(Exception):
        import httpx

        for item in httpx.get(url, timeout=5).json().get("models", []):
            if item.get("name") == model:
                return item.get("context_length")
    return None


MEMORY_KEYS = ("agent_footprint", "runner_footprint", "ollama_rss", "estimate")


class ProcessMemoryPoller:
    """Each second, reads four memory values and keeps the peak of each, in bytes.

    - agent_footprint: the macOS footprint of this process. It includes the Metal memory
      of MLX Whisper.
    - runner_footprint: the macOS footprint of the Ollama model runner.
    - ollama_rss: the RSS sum of all processes with "ollama" in the command.
    - estimate: the Ollama /api/ps size of the loaded models.

    `peak` has the peaks of the full run. `window` has the peaks since the last
    `start_window()`, for example for one call.

    On macOS, the footprint of the llama.cpp runner can miss the mapped model weights. The
    RSS counts only the weight pages that are in memory at that time.
    """

    def __init__(self, api_ps_url: str):
        self.api_ps_url = api_ps_url
        self.peak = dict.fromkeys(MEMORY_KEYS, 0)
        self.window = dict.fromkeys(MEMORY_KEYS, 0)
        self._task: asyncio.Task | None = None

    def start_window(self) -> None:
        self.window = dict.fromkeys(MEMORY_KEYS, 0)

    def record(self, sample: dict) -> None:
        for key, value in sample.items():
            self.peak[key] = max(self.peak[key], value)
            self.window[key] = max(self.window[key], value)

    def _sample(self) -> dict:
        rss, runners = parse_ollama_processes(_ps_all())
        sample = {"ollama_rss": rss,
                  "agent_footprint": _footprint(os.getpid())[0],
                  "runner_footprint": max((_footprint(pid)[0] for pid in runners), default=0)}
        with contextlib.suppress(Exception):
            sample["estimate"] = _api_ps_bytes(self.api_ps_url)
        return sample

    async def _loop(self) -> None:
        while True:
            with contextlib.suppress(OSError, ValueError):
                self.record(await asyncio.to_thread(self._sample))
            await asyncio.sleep(1)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


def memory_upper_gb(memory: dict) -> float:
    """An upper figure for the memory of all models, in GB.

    It adds the agent footprint, the runner footprint, and the /api/ps estimate. The runner
    footprint and the estimate can count the same weights two times, so the sum is high.
    """
    return round(memory.get("agent_footprint_peak_gb", 0.0)
                 + memory.get("ollama_runner_footprint_gb", 0.0)
                 + memory.get("ollama_estimate_gb", 0.0), 1)


def _gb(value: float) -> float:
    return round(value / 1e9, 2)


def agent_peak_rss() -> int:
    """The peak RSS of this process in bytes. macOS gives ru_maxrss in bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# The calls


class RecordingHub(DebugHub):
    """A debug hub that also gives each event to a listener."""

    def __init__(self, listener: Callable[[str, dict], None]):
        super().__init__()
        self._listener = listener

    def publish(self, call_id: str, kind: str, data: dict) -> None:
        super().publish(call_id, kind, data)
        self._listener(kind, data)


@contextlib.contextmanager
def bench_guest_email(email: str):
    """Makes voice guest orders use the benchmark email, as the chat benchmark does.

    A voice guest gives a phone number and no email. The order tool then makes a placeholder
    email from the phone (voice_guest_email). The benchmark changes that email to a
    run-specific address, so that the bench orders are easy to find in the store.
    """
    from support_agent import tools

    original = tools.voice_guest_email
    tools.voice_guest_email = lambda _phone: email
    try:
        yield
    finally:
        tools.voice_guest_email = original


def caller_audio(kokoro, text: str, voice: str, rate: int) -> bytes:
    """Speaks the caller text with Kokoro and returns 16-bit PCM at `rate`.

    An 8 kHz clip is a G.711 fallback test: the resampler limits the band, and there is no
    G.711 codec noise.
    """
    from support_agent.voice.bench.make_audio import resample

    audio, native_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    audio = resample(np.asarray(audio, dtype=np.float32), native_rate, rate)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


# The setup order of a run. The status turns of calls.yaml check it as "{bench_order}".
SETUP_ORDER_HANDLE = "breadboard-830"
SETUP_ORDER_PHONE = "01811111111"


async def place_setup_order(store, email: str) -> int:
    """Places one guest order through the store client, not through the agent.

    The status turns need an order that this run placed, with a phone number that Whisper
    hears correctly. A setup order does not depend on the agent order call, so a failed
    order call does not stop the status calls. The order uses the benchmark email.
    """
    from support_agent.store.models import Address

    product = await store.get_product(SETUP_ORDER_HANDLE)
    if product is None or not product.variants:
        raise RuntimeError(f"The store has no product {SETUP_ORDER_HANDLE}.")
    address = Address(first_name="Karim", last_name="Ali", address_1="House 5 Road 2",
                      city="Sylhet", postal_code="3100", phone=SETUP_ORDER_PHONE)
    order = await store.place_order([(product.variants[0].id, 1)], address, email, None)
    return order.display_id


def check_summary_city(expect: dict, events: list[AgentEvent]) -> list[str]:
    """Checks the city of the last order summary against "summary_city" in the expectation.

    The store client cannot read the address of a placed order back (OrderView has no
    address). So the check reads the address in the last prepare_order result. place_order
    sends that address to the store.
    """
    wanted = expect.get("summary_city")
    if not wanted:
        return []
    city = None
    for e in events:
        if e.type == "tool_end" and e.data.get("name") == "prepare_order" and e.data.get("ok"):
            with contextlib.suppress(KeyError, TypeError, ValueError):
                city = json.loads(e.data["content"])["address"]["city"]
    if city is None:
        return ["no order summary to check the city"]
    if city.strip().lower() != wanted.strip().lower():
        return [f"order summary city: {city.strip()}, the caller said: {wanted}"]
    return []


async def run_call(case, deps, rate: int, voice: str, live: dict, turn_timeout_s: float,
                   bench_order: int | None = None) -> dict:
    """Runs one scripted call through the real pipeline and checks each turn.

    "{bench_order}" in the caller text becomes `bench_order`. If `bench_order` is None, a
    turn that needs it does not play, and the call fails. The result has "placed_order": the number of the order that the call
    placed, or None.
    """
    from pipecat.workers.runner import WorkerRunner

    from support_agent.bench.runner import check_turn
    from support_agent.session import Identity, Session
    from support_agent.voice.pipeline import build_task

    call_failures = []
    script = []
    for turn in case.turns:
        text = fill_caller_text(turn["user"], bench_order)
        if text is None:
            call_failures.append(f"turn not played, no order of this run to check: "
                                 f"{turn['user']}")
        else:
            script.append((turn, text))
    audio = [caller_audio(deps.models.kokoro, text, voice, rate) for _, text in script]
    session = Session.new(Identity.guest(), "voice")
    events: list[list[tuple[str, dict]]] = [[] for _ in script]
    text_times: list[list[float]] = [[] for _ in script]
    answers = [0 for _ in script]
    transport: FileAudioTransport | None = None

    def listen(kind: str, data: dict) -> None:
        record = transport.current if transport is not None else None
        if record is None:
            return
        events[record.index].append((kind, data))
        if kind == "user_text":
            text_times[record.index].append(time.monotonic())
        elif kind in ("answer", "error"):
            answers[record.index] += 1
            no_audio = kind == "error" or not data.get("text", "").strip()
            if no_audio and answer_audio_done(record, text_times[record.index],
                                              answers[record.index]):
                # No answer audio comes, so no stop frame comes after the answer.
                transport.end_turn_when_silent()

    def turn_done(record: TurnRecord) -> bool:
        # The answer event comes before the last answer audio. So the first stop after the
        # last answer ends the answer audio. A stop before it ends a filler or the answer
        # to the first part of a split turn.
        return answer_audio_done(record, text_times[record.index], answers[record.index])

    hub = RecordingHub(listen)
    call_id = hub.start_call("browser" if rate == 16000 else "phone", "bench")
    transport = FileAudioTransport(audio, rate, turn_timeout_s=turn_timeout_s,
                                   turn_done=turn_done)
    worker = build_task(transport, session, deps, hub, call_id, rate)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    started = time.monotonic()
    limit_s = len(script) * (turn_timeout_s + 30) + 30
    try:
        await asyncio.wait_for(runner.run(), limit_s)
    except TimeoutError:
        call_failures.append(f"the call did not end in {limit_s} s")
        await worker.cancel(reason="The benchmark call took too long.")
    seconds = time.monotonic() - started

    turns = []
    for number, (turn, text) in enumerate(script, start=1):
        record = transport.records[number - 1] if number <= len(transport.records) else None
        turn_events = events[number - 1]
        agent_events = [AgentEvent(kind, float(data.get("t_ms", 0.0)), data)
                        for kind, data in turn_events if kind in AGENT_EVENT_TYPES]
        expect = turn.get("expect", {})
        if record is None:
            failures = ["turn not played"]
        elif record.timed_out:
            failures = [f"turn timed out after {turn_timeout_s} s"]
        else:
            failures = check_turn(expect, answer_with_digits(keep_last_answer(agent_events)),
                                  live)
            failures += check_summary_city(expect, agent_events)
            last_answer = next((e for e in reversed(agent_events) if e.type == "answer"), None)
            answer_text = last_answer.data.get("text", "").strip() if last_answer else None
            # The fallback line replaces an empty model answer. It still counts as empty.
            if answer_text is not None and answer_text in ("", EMPTY_ANSWER_FALLBACK):
                failures.append("the agent gave an empty answer")
        had_filler = any(kind == "filler" for kind, _ in turn_events)
        metrics = (turn_metrics(record, had_filler) if record is not None
                   else {"first_audio_s": None, "filler_s": None, "answer_s": None,
                         "talked_over": False})
        agent_turns = sum(1 for e in agent_events if e.type == "turn_start")
        timer_pause = next((d["ms"] for k, d in turn_events if k == "pause"), None)
        turns.append({
            "call": case.id,
            "turn": number,
            "said": text,
            "heard": " ".join(d["text"] for k, d in turn_events if k == "user_text"),
            "tools": [e.data["name"] for e in agent_events if e.type == "tool_start"],
            **metrics,
            "timer_pause_ms": timer_pause,
            "agent_turns": agent_turns,
            "completion_tokens": [e.data.get("tokens") for e in agent_events
                                  if e.type == "llm_end"],
            # A split turn gave more than one agent turn, or the agent spoke over the caller.
            "split": agent_turns > 1 or metrics["talked_over"],
            "answer": " / ".join(e.data["text"] for e in agent_events if e.type == "answer"),
            "agent_audio_s": (round(len(record.agent_audio) / (rate * SAMPLE_BYTES), 1)
                              if record is not None else 0.0),
            "failures": failures,
        })
    failures = call_failures + [f"turn {t['turn']}: {f}" for t in turns for f in t["failures"]]
    all_events = [AgentEvent(kind, 0.0, data) for turn_events in events
                  for kind, data in turn_events if kind == "tool_end"]
    return {"id": case.id, "passed": not failures, "failures": failures, "turns": turns,
            "seconds": round(seconds, 1), "placed_order": placed_order_number(all_events)}


async def _warm_up_llm(settings) -> None:
    """Loads the model in Ollama, so that the first call does not include the model load."""
    import httpx

    url = settings.llm_base_url.removesuffix("/v1") + "/api/generate"
    async with httpx.AsyncClient(timeout=120) as http:
        with contextlib.suppress(httpx.HTTPError):
            await http.post(url, json={"model": settings.llm_model, "prompt": "",
                                       "keep_alive": "30m"})


async def run_calls(rate: int, calls_path: Path = CALLS_PATH,
                    turn_timeout_s: float = TURN_TIMEOUT_S) -> dict:
    from support_agent.bench.runner import _live_prices, load_cases
    from support_agent.config import get_settings
    from support_agent.llm import OpenAiCompatClient
    from support_agent.store.client import StoreClient
    from support_agent.voice.deps import VoiceDeps

    settings = get_settings()
    store = StoreClient(settings)
    deps = VoiceDeps(settings=settings, store=store, llm=OpenAiCompatClient(settings))
    voice = next(v for v in CALLER_VOICES if v != settings.voice_tts_voice)
    cases = load_cases(str(calls_path))
    run_id = uuid.uuid4().hex[:8]
    poller = ProcessMemoryPoller(settings.llm_base_url.removesuffix("/v1") + "/api/ps")
    calls = []
    try:
        live = await _live_prices(store, cases)
        await deps.models.ensure_loaded()
        await _warm_up_llm(settings)
        poller.start()
        context_length = await asyncio.to_thread(_context_length, poller.api_ps_url,
                                                 settings.llm_model)
        started = time.monotonic()
        wall_started = time.time()
        email = f"bench-{run_id}@example.com"
        bench_order = await place_setup_order(store, email)
        with bench_guest_email(email):
            for case in cases:
                poller.start_window()
                call = await run_call(case, deps, rate, voice, live, turn_timeout_s,
                                      bench_order)
                window = dict(poller.window)
                call["memory"] = {
                    "agent_footprint_gb": _gb(window["agent_footprint"]),
                    "runner_footprint_gb": _gb(window["runner_footprint"]),
                    "ollama_rss_gb": _gb(window["ollama_rss"]),
                    "estimate_gb": _gb(window["estimate"]),
                }
                print(f"{'pass' if call['passed'] else 'FAIL'}  {case.id}  "
                      f"{'; '.join(call['failures'])}", flush=True)
                calls.append(call)
        seconds = round(time.monotonic() - started, 1)
        slept = slept_seconds(time.time() - wall_started, time.monotonic() - started)
    finally:
        await poller.stop()
        await store.aclose()
    agent_lifetime_peak = max(_footprint(os.getpid()))
    return {
        "rate": rate,
        "model": settings.llm_model,
        "reasoning_effort": settings.llm_reasoning_effort,
        "reasoning_effort_after_tools": settings.llm_reasoning_effort_after_tools,
        "max_prompt_tokens": settings.max_prompt_tokens,
        "stt_model": settings.voice_stt_model,
        "agent_voice": settings.voice_tts_voice,
        "caller_voice": voice,
        "run_id": run_id,
        "setup_order": bench_order,
        "context_length": context_length,
        "slept_s": slept,
        "seconds": seconds,
        "calls": calls,
        "memory": {
            "agent_peak_rss_gb": _gb(agent_peak_rss()),
            "agent_footprint_peak_gb": _gb(max(agent_lifetime_peak,
                                               poller.peak["agent_footprint"])),
            "ollama_peak_rss_gb": _gb(poller.peak["ollama_rss"]),
            "ollama_runner_footprint_gb": _gb(poller.peak["runner_footprint"]),
            "ollama_estimate_gb": _gb(poller.peak["estimate"]),
        },
    }


# The report

# Goals of the voice design. A missed goal does not fail the run.
PAUSE_GOAL_S = 1.5
FILLER_GOAL_S = 1.5
MEMORY_GOAL_GB = 12.0
# The evidence of the split-turn defect from the first Task 9 run (16 kHz, run 69f1cb75).
SPLIT_TURN_EVIDENCE = [
    "In run 69f1cb75 (16 kHz), the caller said \"My name is Karim Ali. The address is House 5,",
    "Road 2, Sylhet, postal code 3100. My phone number is 018 1111 1111.\" in one turn. The",
    "turn-end model ended the turn after the first sentence. The agent answered while the",
    "caller still spoke. The answer audio started 4.27 s before the end of the caller audio",
    "(a pause of -4.27 s).",
]


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _median(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _stats(values: list[float | None]) -> tuple[str, str, int]:
    present = [v for v in values if v is not None]
    if not present:
        return "-", "-", 0
    return (f"{statistics.median(present):.2f}", f"{percentile(present, 0.9):.2f}",
            len(present))


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _verdict(value: float | None, goal: float) -> str:
    if value is None:
        return "no data"
    return "met" if value <= goal else "missed"


def _goal_lines(turns: list[dict], calls: list[dict], memory: dict) -> list[str]:
    pause = _median([t["first_audio_s"] for t in turns if not t["tools"]])
    filler = _median([t["filler_s"] for t in turns if t["tools"]])
    upper = memory_upper_gb(memory)
    passed = sum(c["passed"] for c in calls)
    return [
        "## Goals",
        "",
        "Each goal is a goal, not a gate. A missed goal does not fail the run.",
        "",
        "| Goal | Target | Result | Verdict |",
        "|---|---|---|---|",
        (f"| Pause median, turns without tools | {PAUSE_GOAL_S} s or less | "
         f"{_seconds(pause)} s | {_verdict(pause, PAUSE_GOAL_S)} |"),
        (f"| Filler start median, turns with tools | {FILLER_GOAL_S} s or less | "
         f"{_seconds(filler)} s | {_verdict(filler, FILLER_GOAL_S)} |"),
        (f"| Memory upper figure | {MEMORY_GOAL_GB:.0f} GB or less | {upper:.1f} GB | "
         f"{_verdict(upper, MEMORY_GOAL_GB)} |"),
        (f"| Calls passed | all | {passed} of {len(calls)} | "
         f"{'met' if passed == len(calls) else 'missed'} |"),
        "",
        "The memory upper figure is the agent footprint, plus the runner footprint, plus the",
        "Ollama /api/ps estimate. The last two can count the same weights two times.",
        "",
    ]


def render_report(result: dict, day: str | None = None) -> str:
    day = day or result.get("date", "")
    rate = result["rate"]
    rate_label = (f"{rate} Hz (the main path: browser and G.722 phone)" if rate == 16000
                  else f"{rate} Hz (G.711 fallback, band-limit only, no codec noise)")
    calls = result["calls"]
    turns = [t for call in calls for t in call["turns"]]
    split = [t for t in turns if t.get("split")]
    talked_over = [t for t in turns if t.get("talked_over")]
    with_tools = [t for t in turns if t["tools"]]
    without_tools = [t for t in turns if not t["tools"]]
    memory = result["memory"]
    lines = [
        f"# Voice call benchmark at {rate_label}",
        "",
        f"Date: {day}. Model: {result['model']}"
        + (f" (reasoning {result['reasoning_effort']})" if result.get("reasoning_effort") else "")
        + (f", reasoning after tools {result['reasoning_effort_after_tools']}"
           if result.get("reasoning_effort_after_tools") else "")
        + (f", prompt budget {result['max_prompt_tokens']} estimated tokens"
           if result.get("max_prompt_tokens") else "")
        + f". Speech-to-text: {result['stt_model']}. Run ID: {result['run_id']}.",
        (f"Agent voice: {result.get('agent_voice', '-')}. "
         f"Caller voice: {result.get('caller_voice', '-')}. Run time: {result['seconds']} s."),
        (f"Ollama context: {result.get('context_length', '-')} tokens. "
         f"Setup order: {result.get('setup_order', '-')}. The bench placed the setup order "
         "through the store client for the status turns."),
        "",
        *([(f"NOT VALID: the computer slept for {result['slept_s']:.0f} s during the run. "
           "The times and the timeouts are wrong. Run it again with caffeinate -i."), ""]
          if result.get("slept_s") else []),
        (f"Sample: {len(calls)} calls, {len(turns)} turns. This is a single run, rough "
         "baseline. Run it more than one time before you compare two changes."),
        "",
        "The benchmark plays the caller turns of calls.yaml as Kokoro audio through the real",
        "pipeline (VAD, smart turn, Whisper, agent, Kokoro). FileAudioTransport sends the caller",
        "audio in real time and waits until the answer audio of each turn ends.",
        "All times are from the end of the caller audio, so they include the VAD stop time and",
        "the smart-turn decision. The user chat server on port 8100 ran during the run. A user",
        "test call at the same time can make a turn slow.",
        "",
    ]
    lines += _goal_lines(turns, calls, memory)
    lines += [
        "## Pause",
        "",
        "The pause is the time to the first agent audio after the end of the caller audio. In",
        "a turn with tools, the first agent audio is the filler. The answer start is the time",
        "to the answer audio.",
        "",
        (f"Split turns: {len(split)}. A split turn gave more than one agent turn, or the agent"
         " spoke before the caller audio ended. The rows without tools, with tools, and all"
         " turns include the split turns. The split turns also have their own row."
         f" Turns where the agent spoke over the caller: {len(talked_over)}."),
        "",
        ("| Turns | Count | Pause median (s) | Pause p90 (s) | Answer median (s) | "
         "Answer p90 (s) |"),
        "|---|---|---|---|---|---|",
    ]
    for label, group in (("Without tools", without_tools), ("With tools", with_tools),
                         ("Split turns", split), ("All turns", turns)):
        pause_median, pause_p90, _ = _stats([t["first_audio_s"] for t in group])
        answer_median, answer_p90, _ = _stats([t["answer_s"] for t in group])
        lines.append(f"| {label} | {len(group)} | {pause_median} | {pause_p90} | "
                     f"{answer_median} | {answer_p90} |")
    lines += [
        "",
        "| Call | Turn | Tools | Pause (s) | Answer (s) | Timer pause (ms) | Split | Heard |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in turns:
        lines.append(
            f"| {t['call']} | {t['turn']} | {', '.join(t['tools']) or '-'} | "
            f"{_seconds(t['first_audio_s'])} | {_seconds(t['answer_s'])} | "
            f"{t['timer_pause_ms'] if t['timer_pause_ms'] is not None else '-'} | "
            f"{'yes' if t.get('split') else 'no'} | {_cell(t['heard'])} |")
    lines += [
        "",
        "The timer pause is the TurnTimer value: the VAD stop to the first TTS audio frame.",
        "",
        "## Filler",
        "",
    ]
    fillers = [t for t in turns if t["filler_s"] is not None]
    filler_median, filler_p90, count = _stats([t["filler_s"] for t in fillers])
    lines += [
        (f"Turns with a filler: {count}. Filler start median: {filler_median} s. "
         f"Filler start p90: {filler_p90} s."),
        "",
        "| Call | Turn | Filler start (s) | Answer start (s) | Gap (s) |",
        "|---|---|---|---|---|",
    ]
    for t in fillers:
        gap = (None if t["answer_s"] is None else round(t["answer_s"] - t["filler_s"], 3))
        lines.append(f"| {t['call']} | {t['turn']} | {_seconds(t['filler_s'])} | "
                     f"{_seconds(t['answer_s'])} | {_seconds(gap)} |")
    passed = sum(c["passed"] for c in calls)
    lines += [
        "",
        "## Calls",
        "",
        f"Passed: {passed} of {len(calls)}. The checks are the chat benchmark checks.",
        "The benchmark changes number words in the answer to digits before the checks.",
        "A number in a check does not match a longer number: \"270\" does not match \"2700\".",
        "",
        "| Call | Result | Failures |",
        "|---|---|---|",
    ]
    for call in calls:
        lines.append(f"| {call['id']} | {'pass' if call['passed'] else 'FAIL'} | "
                     f"{_cell('; '.join(call['failures'])) or '-'} |")
    lines += ["", "| Call | Turn | Said | Answer |", "|---|---|---|---|"]
    for t in turns:
        lines.append(f"| {t['call']} | {t['turn']} | {_cell(t.get('said', ''))} | "
                     f"{_cell(t['answer'])} |")
    lines += [
        "",
        "## Memory",
        "",
        (f"Memory upper figure: {memory_upper_gb(memory):.1f} GB. The goal is "
         f"{MEMORY_GOAL_GB:.0f} GB or less."),
        "",
        "| Process | Peak of the run (GB) |",
        "|---|---|",
        f"| Agent process, RSS | {memory['agent_peak_rss_gb']} |",
    ]
    if "agent_footprint_peak_gb" in memory:
        lines.append(f"| Agent process, macOS footprint | {memory['agent_footprint_peak_gb']} |")
    lines.append(f"| Ollama processes, RSS sum | {memory['ollama_peak_rss_gb']} |")
    lines.append(f"| Ollama runner, macOS footprint | {memory['ollama_runner_footprint_gb']} |")
    if "ollama_estimate_gb" in memory:
        lines.append(f"| Ollama /api/ps estimate | {memory['ollama_estimate_gb']} |")
    with_memory = [c for c in calls if c.get("memory")]
    if with_memory:
        lines += [
            "",
            "Peaks during each call (GB). The upper figure adds the agent footprint, the runner",
            "footprint, and the estimate.",
            "",
            ("| Call | Turns | Agent footprint | Runner footprint | Ollama RSS | Estimate | "
             "Upper figure |"),
            "|---|---|---|---|---|---|---|",
        ]
        for c in with_memory:
            m = c["memory"]
            upper = round(m["agent_footprint_gb"] + m["runner_footprint_gb"] + m["estimate_gb"],
                          1)
            lines.append(f"| {c['id']} | {len(c['turns'])} | {m['agent_footprint_gb']} | "
                         f"{m['runner_footprint_gb']} | {m['ollama_rss_gb']} | "
                         f"{m['estimate_gb']} | {upper} |")
    lines += [
        "",
        "On macOS, the footprint of the llama.cpp runner can miss the mapped model weights,",
        "as the model comparison report found. The RSS counts only the weight pages that are",
        "in memory at that time. Use the Ollama numbers as a lower limit, not as the full",
        "memory of the model. The /api/ps estimate is the Ollama estimate of the weights and",
        "the context. The agent footprint includes the Metal memory of MLX Whisper. The agent",
        "RSS can miss that memory. The Ollama RSS sum includes the Ollama app and server.",
        "",
        "## Known findings",
        "",
        "Split turns: the turn-end model (smart turn) ends a caller turn at a pause inside a",
        "sentence. Then the agent answers the first part and speaks over the caller.",
        *SPLIT_TURN_EVIDENCE,
        (f"In this run: {len(split)} split turns, and {len(talked_over)} turns where the agent"
         " spoke over the caller."),
        "",
        "The order city: the store client cannot read the address of a placed order back,",
        "because OrderView has no address. So the city check reads the address in the last",
        "prepare_order result. place_order sends that address to the store.",
        "",
    ]
    empty = [f for c in calls for f in c["failures"] if f.endswith("the agent gave an empty answer")]
    lines += [
        (f"Empty answers: {len(empty)}. The count includes the turns where the agent said the "
         f"fallback line \"{EMPTY_ANSWER_FALLBACK}\" in place of an empty model answer. The "
         f"Ollama context is {result.get('context_length', '-')} tokens. The agent keeps the "
         f"prompt under {result.get('max_prompt_tokens', '-')} estimated tokens."),
        "",
    ]
    facts = [f for c in calls for f in c["failures"]
             if "order summary city" in f or "does not have" in f]
    if facts:
        lines += ["Wrong spoken or stored facts in this run:", ""]
        lines += [f"- {c['id']}: {f}" for c in calls for f in c["failures"] if f in facts]
        lines.append("")
    lines += [
        "## Note",
        "",
        "The caller audio is synthetic speech with no noise. A real caller gives more",
        "speech-to-text errors. The 8 kHz run is a G.711 fallback test only. The main",
        "path is 16 kHz.",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict, reports_dir: Path = REPORTS_DIR) -> Path:
    """Writes the report and the raw result JSON next to it. Returns the report path."""
    result.setdefault("date", date.today().isoformat())
    path = reports_dir / f"{result['date']}-voice-calls-{result['rate']}.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    path.write_text(render_report(result))
    return path


def rebuild_report(json_path: Path) -> Path:
    """Renders the report again from a raw result JSON, with no run. Returns the report path."""
    result = json.loads(Path(json_path).read_text())
    path = Path(json_path).with_suffix(".md")
    path.write_text(render_report(result))
    return path
