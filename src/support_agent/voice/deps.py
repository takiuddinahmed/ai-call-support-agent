"""Light parts of the voice channel that the server imports at start.

This module must not import Pipecat or a speech model package. The server imports it at
start, and the chat server must start as fast as before.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from support_agent.config import Settings
from support_agent.llm import LlmClient

KOKORO_MODEL_NAME = "kokoro-v1.0.onnx"
KOKORO_VOICES_NAME = "voices-v1.0.bin"


class VoiceModels:
    """Loads the Whisper and Kokoro models at the first call and keeps them loaded."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = asyncio.Lock()
        self.kokoro = None

    @property
    def loaded(self) -> bool:
        return self.kokoro is not None

    async def ensure_loaded(self) -> None:
        async with self._lock:
            if not self.loaded:
                await asyncio.to_thread(self._load)

    def _load(self) -> None:
        import mlx_whisper
        import numpy as np
        from kokoro_onnx import Kokoro
        from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

        logger.info("Loading the voice models. The first download can take a few minutes.")
        model = Path(KOKORO_CACHE_DIR) / KOKORO_MODEL_NAME
        voices = Path(KOKORO_CACHE_DIR) / KOKORO_VOICES_NAME
        _ensure_model_files(model, voices)
        kokoro = Kokoro(str(model), str(voices))
        # mlx_whisper keeps the last model in memory. One short transcription loads it.
        mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32),
                               path_or_hf_repo=self._settings.voice_stt_model, language="en")
        self.kokoro = kokoro
        logger.info("The voice models are loaded.")


@dataclass
class VoiceDeps:
    settings: Settings
    store: object
    llm: LlmClient
    models: VoiceModels = field(init=False)

    def __post_init__(self):
        self.models = VoiceModels(self.settings)
