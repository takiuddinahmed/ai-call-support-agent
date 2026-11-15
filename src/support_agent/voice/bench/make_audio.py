"""Makes the benchmark audio files with Kokoro voices and macOS `say` voices."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

AUDIO_DIR = Path("bench-audio")
# 16 kHz is the main phone path (Asterisk slin16). 8 kHz is a G.711 fallback test that
# has a band-limit only and no codec noise.
RATES = (16000, 8000)
KOKORO_VOICES = ("af_heart", "am_adam")
SAY_VOICES = ("Samantha", "Daniel")
SENTENCES_PATH = Path(__file__).with_name("sentences.yaml")


def load_sentences() -> list[dict]:
    """Returns the sentences. Each one has `group` and `text`."""
    return yaml.safe_load(SENTENCES_PATH.read_text())


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resamples mono float audio. Returns the input if the rates are equal."""
    if from_rate == to_rate:
        return audio.astype(np.float32)
    divisor = np.gcd(from_rate, to_rate)
    out = resample_poly(audio, to_rate // divisor, from_rate // divisor)
    return out.astype(np.float32)


def audio_path(voice: str, index: int, rate: int, audio_dir: Path = AUDIO_DIR) -> Path:
    return audio_dir / f"{voice}-{index:02d}-{rate}.wav"


def _say(voice: str, text: str) -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "say.wav"
        subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@22050", text],
                       check=True)
        audio, rate = sf.read(path, dtype="float32")
    return audio, rate


def make_audio(audio_dir: Path = AUDIO_DIR) -> int:
    """Writes one WAV file for each voice, sentence, and rate. Returns the file count."""
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

    from support_agent.voice.deps import KOKORO_MODEL_NAME, KOKORO_VOICES_NAME

    model = Path(KOKORO_CACHE_DIR) / KOKORO_MODEL_NAME
    voices = Path(KOKORO_CACHE_DIR) / KOKORO_VOICES_NAME
    _ensure_model_files(model, voices)
    kokoro = Kokoro(str(model), str(voices))

    audio_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, sentence in enumerate(load_sentences()):
        text = sentence["text"]
        clips = {}
        for voice in KOKORO_VOICES:
            clips[voice] = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        for voice in SAY_VOICES:
            clips[voice] = _say(voice, text)
        for voice, (audio, native_rate) in clips.items():
            for rate in RATES:
                sf.write(audio_path(voice, index, rate, audio_dir),
                         resample(audio, native_rate, rate), rate)
                count += 1
    return count
