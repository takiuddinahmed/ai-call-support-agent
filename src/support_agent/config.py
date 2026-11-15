"""Configuration of the agent. Values come from the environment or from .env."""

import shlex
import subprocess
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    store_url: str = "http://localhost:9000"
    store_publishable_key: str = ""
    # Reads the key that the store seed script writes. Run the agent from agent/.
    store_key_file_command: str = (
        "docker compose -f ../demo-store/docker-compose.yml exec -T medusa "
        "cat /shared/publishable_key"
    )
    llm_base_url: str = "http://localhost:11434/v1"
    # Chosen in reports/2026-11-15-model-comparison.md.
    llm_model: str = "qwen3.5:4b-q4_K_M"
    llm_api_key: str = "ollama"
    # Hidden thinking of the model: none, low, medium, or high. Empty sends nothing.
    # Measured with qwen3.5:4b: the model default thinks for about 18 s; "none" answers in
    # about 0.2 s but skips the store tools and invents prices; "low" uses the tools and
    # gives the first token in about 2 s.
    llm_reasoning_effort: str = "low"
    max_tool_rounds: int = 6
    catalog_cache_seconds: int = 300
    # Voice channel. The speech benchmark can change the models.
    voice_stt_model: str = "mlx-community/whisper-small-mlx"
    voice_tts_voice: str = "af_heart"
    voice_asterisk_path: str = "/api/voice/asterisk"
    # Shared secret of the phone route (env AGENT_PHONE_TOKEN). Asterisk sends it as the
    # HTTP Basic password ("password" in websocket_client.conf). If it is empty, the
    # phone route refuses every call.
    phone_token: str = Field("", validation_alias=AliasChoices("AGENT_PHONE_TOKEN",
                                                                "phone_token"))
    # The phone route ends a call after this time, in seconds.
    max_call_seconds: int = 900
    # Host names that the server accepts in the Host header. Unknown names get 400.
    # Asterisk in Docker connects to the phone route with host.docker.internal. That name gets
    # 403 on every route except the phone route. The debug routes accept only local clients.
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "host.docker.internal"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_publishable_key(settings: Settings) -> str:
    """Returns the store publishable key. If the setting is empty, runs the command."""
    if settings.store_publishable_key:
        return settings.store_publishable_key
    result = subprocess.run(
        shlex.split(settings.store_key_file_command),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()
