"""Configuration of the agent. Values come from the environment or from .env."""

import shlex
import subprocess
from functools import lru_cache

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
    llm_model: str = "qwen3.5:4b"
    llm_api_key: str = "ollama"
    # Hidden thinking of the model: none, low, medium, or high. Empty sends nothing.
    # Measured with qwen3.5:4b: the model default thinks for about 18 s; "none" answers in
    # about 0.2 s but skips the store tools and invents prices; "low" uses the tools and
    # gives the first token in about 2 s.
    llm_reasoning_effort: str = "low"
    max_tool_rounds: int = 6
    catalog_cache_seconds: int = 300


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
