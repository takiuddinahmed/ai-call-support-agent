"""Benchmark: runs scripted conversations and reports correctness, speed, and memory."""

import argparse
import asyncio
import glob
import json
import re
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import yaml

from support_agent.config import Settings, get_settings
from support_agent.llm import OpenAiCompatClient
from support_agent.loop import AgentEvent, run_turn
from support_agent.session import Identity, Session
from support_agent.store.client import StoreClient
from support_agent.tools import ToolRunner

CASES_GLOB = str(Path(__file__).parent / "cases" / "*.yaml")
BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
LIVE_PRICE = re.compile(r"\{live:price:([a-z0-9-]+)\}")
LOGINS = {"rahim": ("rahim@example.com", "Demo1234!")}


@dataclass
class Case:
    id: str
    language: str
    login: str
    channel: str
    turns: list[dict]


@dataclass
class CaseResult:
    id: str
    language: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)


def load_cases(pattern: str = CASES_GLOB) -> list[Case]:
    cases = []
    for path in sorted(glob.glob(pattern)):
        for item in yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []:
            cases.append(Case(**item))
    return cases


def normalize_answer(text: str) -> str:
    """Lower case, ASCII digits, and no thousands commas, so that prices compare."""
    text = text.translate(BANGLA_DIGITS).lower()
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def _expand(value: str, live: dict[str, int]) -> str:
    return LIVE_PRICE.sub(lambda m: str(live.get(m.group(1), "?")), value)


def check_turn(expect: dict, events: list[AgentEvent], live: dict[str, int]) -> list[str]:
    failures = [f"error: {e.data['message']}" for e in events if e.type == "error"]
    called = [e.data["name"] for e in events if e.type == "tool_start"]
    called += [e.data["name"] for e in events if e.type == "tool_end" and e.data["name"] not in called]
    answer = next((e.data["text"] for e in events if e.type == "answer"), "")
    text = normalize_answer(answer)
    for name in expect.get("tools_called", []):
        if name not in called:
            failures.append(f"tool not called: {name}")
    for name in expect.get("tools_not_called", []):
        if name in called:
            failures.append(f"tool called but must not be: {name}")
    wanted = [normalize_answer(_expand(w, live)) for w in expect.get("answer_contains_any", [])]
    if wanted and not any(w in text for w in wanted):
        failures.append(f"answer has none of: {wanted}")
    for bad in expect.get("answer_not_contains", []):
        if normalize_answer(_expand(bad, live)) in text:
            failures.append(f"answer contains: {bad}")
    if "order_placed" in expect:
        placed = any(e.type == "tool_end" and e.data["name"] == "place_order" and e.data["ok"]
                     for e in events)
        if placed != expect["order_placed"]:
            failures.append(f"order placed: {placed}, expected: {expect['order_placed']}")
    if expect.get("no_list_lines"):
        if re.search(r"^\s*(-|\*|\d+\.)\s", answer, re.MULTILINE):
            failures.append("answer has list lines")
    return failures


async def _live_prices(store, cases: list[Case]) -> dict[str, int]:
    handles = set()
    for case in cases:
        for turn in case.turns:
            for value in json.dumps(turn.get("expect", {})).split('"'):
                handles.update(LIVE_PRICE.findall(value))
    prices = {}
    for handle in handles:
        product = await store.get_product(handle)
        if product:
            prices[handle] = product.lowest_price_bdt
    return prices


async def run_case(case: Case, model: str, store, llm, max_tool_rounds: int, run_id: str,
                   live: dict[str, int] | None = None) -> CaseResult:
    live = live if live is not None else await _live_prices(store, [case])
    identity = Identity.guest()
    if case.login in LOGINS:
        token = await store.login(*LOGINS[case.login])
        identity = Identity(kind="customer", token=token, customer=await store.get_customer(token))
    session = Session.new(identity, case.channel)
    result = CaseResult(case.id, case.language, True)
    for number, turn in enumerate(case.turns, start=1):
        text = turn["user"].replace("{bench_email}", f"bench-{run_id}@example.com")
        events = [e async for e in run_turn(session, text, llm, ToolRunner(store, session), model,
                                            max_tool_rounds)]
        failures = check_turn(turn.get("expect", {}), events, live)
        first = next((e.data["ms"] for e in events if e.type == "first_token"), None)
        answer = next((e.data["text"] for e in events if e.type == "answer"), "")
        result.turns.append({
            "turn": number,
            "first_token_ms": first,
            "total_ms": events[-1].t_ms if events else None,
            "tools": [e.data["name"] for e in events if e.type == "tool_start"],
            "tokens_per_s": [e.data["tokens_per_s"] for e in events
                             if e.type == "llm_end" and e.data.get("tokens_per_s")],
            "answer": answer,
            "failures": failures,
        })
        if failures:
            result.passed = False
            result.failures += [f"turn {number}: {f}" for f in failures]
    return result


def parse_footprint(text: str) -> tuple[int, int]:
    """Reads phys_footprint and phys_footprint_peak (bytes) from macOS `footprint -f bytes`."""
    now = re.search(r"phys_footprint:\s+(\d+) B", text)
    peak = re.search(r"phys_footprint_peak:\s+(\d+) B", text)
    return (int(now.group(1)) if now else 0, int(peak.group(1)) if peak else 0)


async def _runner_footprint(model: str) -> tuple[int, int]:
    """Real memory of the Ollama runner process for the model, GPU memory included (macOS)."""
    find = await asyncio.create_subprocess_exec(
        "pgrep", "-f", f"ollama runner --model {model}", stdout=asyncio.subprocess.PIPE
    )
    pids = (await find.communicate())[0].decode().split()
    if not pids:
        return 0, 0
    proc = await asyncio.create_subprocess_exec(
        "footprint", "-f", "bytes", pids[0],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    return parse_footprint((await proc.communicate())[0].decode())


class MemoryPoller:
    """Every second, reads the Ollama /api/ps size (an estimate) and the real footprint of
    the runner process, and keeps the peak of each."""

    def __init__(self, base_url: str, model: str = ""):
        self.url = base_url.removesuffix("/v1") + "/api/ps"
        self.model = model
        self.peak_bytes = 0
        self.peak_footprint = 0
        self._task: asyncio.Task | None = None

    async def _loop(self):
        async with httpx.AsyncClient(timeout=5) as http:
            while True:
                try:
                    data = (await http.get(self.url)).json()
                    size = sum(m.get("size", 0) for m in data.get("models", []))
                    self.peak_bytes = max(self.peak_bytes, size)
                except (httpx.HTTPError, ValueError):
                    pass
                if self.model:
                    try:
                        now, peak = await _runner_footprint(self.model)
                        self.peak_footprint = max(self.peak_footprint, now, peak)
                    except OSError:
                        pass
                await asyncio.sleep(1)

    def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()


def _percentile(values: list[float], p: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return round(values[min(len(values) - 1, int(round(p * (len(values) - 1))))], 1)


def summarize(model: str, results: list[CaseResult], peak_bytes: int, seconds: float) -> dict:
    turns = [t for r in results for t in r.turns]
    firsts = [t["first_token_ms"] for t in turns]
    totals = [t["total_ms"] for t in turns]
    speeds = [s for t in turns for s in t["tokens_per_s"]]
    by_language = {}
    for lang in sorted({r.language for r in results}):
        group = [r for r in results if r.language == lang]
        by_language[lang] = f"{sum(r.passed for r in group)}/{len(group)}"
    return {
        "model": model,
        "date": datetime.now().isoformat(timespec="seconds"),
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(100 * sum(r.passed for r in results) / max(len(results), 1), 1),
        "by_language": by_language,
        "first_token_ms_median": _percentile(firsts, 0.5),
        "first_token_ms_p90": _percentile(firsts, 0.9),
        "turn_ms_median": _percentile(totals, 0.5),
        "turn_ms_p90": _percentile(totals, 0.9),
        "tokens_per_s_median": round(statistics.median(speeds), 1) if speeds else None,
        "peak_model_memory_gb": round(peak_bytes / 1e9, 2),
        "run_seconds": round(seconds, 1),
    }


def write_report(out: Path, summary: dict, results: list[CaseResult]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y-%m-%d-%H%M}-{summary['model'].replace(':', '_').replace('/', '_')}"
    (out / f"{stem}.json").write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, ensure_ascii=False,
                   indent=2), encoding="utf-8")
    lines = [f"# Benchmark: {summary['model']}", "", "| Measure | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    lines += ["", "| Case | Language | Result | Failures |", "| --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r.id} | {r.language} | {'pass' if r.passed else 'FAIL'} | "
                     f"{'; '.join(r.failures).replace('|', '/')} |")
    path = out / f"{stem}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def main_async(args) -> int:
    settings: Settings = get_settings()
    if args.reasoning is not None:
        settings = settings.model_copy(update={"llm_reasoning_effort": args.reasoning})
    store = StoreClient(settings)
    llm = OpenAiCompatClient(settings)
    cases = load_cases(args.cases)
    run_id = uuid.uuid4().hex[:8]
    live = await _live_prices(store, cases)
    poller = MemoryPoller(settings.llm_base_url, args.model)
    poller.start()
    start = time.perf_counter()
    results = []
    for case in cases:
        result = await run_case(case, args.model, store, llm, settings.max_tool_rounds, run_id, live)
        print(f"{'pass' if result.passed else 'FAIL'}  {case.id}  {'; '.join(result.failures)}")
        results.append(result)
    await poller.stop()
    await store.aclose()
    summary = summarize(args.model, results, poller.peak_bytes, time.perf_counter() - start)
    summary["reasoning_effort"] = settings.llm_reasoning_effort
    summary["peak_runner_footprint_gb"] = round(poller.peak_footprint / 1e9, 2)
    path = write_report(Path(args.out), summary, results)
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the support agent benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", default=CASES_GLOB)
    parser.add_argument("--out", default="reports")
    parser.add_argument("--reasoning", default=None, help="Override LLM_REASONING_EFFORT.")
    return asyncio.run(main_async(parser.parse_args()))
