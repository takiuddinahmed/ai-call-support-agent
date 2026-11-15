"""Speech-to-text benchmark: word error rate and time of one Whisper model."""

import statistics
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

from support_agent.voice.bench.make_audio import (
    AUDIO_DIR,
    KOKORO_VOICES,
    SAY_VOICES,
    audio_path,
    load_sentences,
    resample,
)
from support_agent.voice.bench.wer import normalize_for_wer, word_error_rate

WHISPER_RATE = 16000
REPORTS_DIR = Path("reports")


def percentile(values: list[float], fraction: float) -> float:
    """Returns the value at `fraction` of the sorted list, with linear interpolation."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def load_clip(path: Path) -> np.ndarray:
    """Reads a WAV file and resamples it to the 16 kHz that Whisper needs.

    An 8 kHz clip is a G.711 fallback test. The 8 kHz clip is a band-limit only (the
    resampler removes the high band) and has no G.711 codec noise. The main phone path
    is 16 kHz audio from Asterisk (slin16, G.722 phones), which needs no resampling.
    """
    audio, rate = sf.read(path, dtype="float32")
    return resample(audio, rate, WHISPER_RATE)


def run_stt(model: str, rate: int, audio_dir: Path = AUDIO_DIR) -> dict:
    """Transcribes every clip and returns the rows, one for each clip."""
    import mlx_whisper

    sentences = load_sentences()
    voices = KOKORO_VOICES + SAY_VOICES
    # The model loads once, in the warm-up call. The timed calls do not include the load.
    mlx_whisper.transcribe(np.zeros(WHISPER_RATE, dtype=np.float32), path_or_hf_repo=model,
                           language="en")
    rows = []
    for index, sentence in enumerate(sentences):
        for voice in voices:
            clip = load_clip(audio_path(voice, index, rate, audio_dir))
            start = time.perf_counter()
            result = mlx_whisper.transcribe(clip, path_or_hf_repo=model, language="en")
            seconds = time.perf_counter() - start
            heard = result["text"].strip()
            rows.append({
                "group": sentence["group"],
                "voice": voice,
                "reference": sentence["text"],
                "heard": heard,
                "wer": word_error_rate(normalize_for_wer(sentence["text"]),
                                       normalize_for_wer(heard)),
                "raw_wer": word_error_rate(sentence["text"], heard),
                "seconds": seconds,
            })
    return {"model": model, "rate": rate, "rows": rows}


def render_report(result: dict, day: str) -> str:
    rows = result["rows"]
    by_group = defaultdict(list)
    by_voice = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
        by_voice[row["voice"]].append(row)

    def mean(items: list[dict], key: str) -> float:
        return statistics.mean(item[key] for item in items)

    rate_label = (f"{result['rate']} Hz"
                  if result["rate"] == 16000
                  else "8000 Hz (G.711 fallback, band-limit only, no codec noise)")
    times = [row["seconds"] for row in rows]
    lines = [
        f"# Speech-to-text benchmark: {result['model']} at {rate_label}",
        "",
        f"Date: {day}. Clips: {len(rows)}. Language: English.",
        "",
        "## Word error rate",
        "",
        "The main number is the normalized word error rate. The normalization makes the",
        "format of numbers and codes the same on both sides (for example ESP32 and ESP 32).",
        "The raw rate counts those format differences as errors.",
        "",
        (f"The mean normalized word error rate is {mean(rows, 'wer'):.3f}. "
         f"The mean raw word error rate is {mean(rows, 'raw_wer'):.3f}."),
        "",
        "| Sentence group | Normalized | Raw | Clips |",
        "|---|---|---|---|",
    ]
    lines += [f"| {g} | {mean(v, 'wer'):.3f} | {mean(v, 'raw_wer'):.3f} | {len(v)} |"
              for g, v in by_group.items()]
    lines += ["", "| Voice | Normalized | Raw | Clips |", "|---|---|---|---|"]
    lines += [f"| {g} | {mean(v, 'wer'):.3f} | {mean(v, 'raw_wer'):.3f} | {len(v)} |"
              for g, v in by_voice.items()]
    lines += [
        "",
        "## Time",
        "",
        "Each Whisper call ran alone, after the model loaded and one warm-up call.",
        "The times are Whisper time only. They do not include audio capture, voice activity",
        "detection, or speech synthesis. Ollama had a model loaded during the timing, so",
        "the times can have a small error.",
        "",
        f"- Median time: {statistics.median(times):.3f} s",
        f"- p90 time: {percentile(times, 0.9):.3f} s",
        "",
        "## Worst clips",
        "",
        "| Word error rate | Voice | Reference | Heard |",
        "|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r["wer"], reverse=True)[:8]:
        lines.append(f"| {row['wer']:.2f} | {row['voice']} | {row['reference']} | "
                     f"{row['heard']} |")
    lines += [
        "",
        "## Note",
        "",
        f"Differences of about 0.01 in word error rate are inside the noise of {len(rows)}",
        "clips. Do not rank models on a difference that small.",
        "",
        "The 8 kHz clips are a G.711 fallback test. They are band-limited only and have no",
        "G.711 codec noise. The main phone path is 16 kHz.",
        "",
        "Synthetic voices give lower error rates than real callers. Real callers have noise,",
        "accents, and a poor phone line. Use these numbers to compare models, not to predict",
        "the error rate of a real call.",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict, reports_dir: Path = REPORTS_DIR) -> Path:
    today = date.today().isoformat()
    name = result["model"].split("/")[-1]
    path = reports_dir / f"{today}-stt-{name}-{result['rate']}.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(result, today))
    return path
