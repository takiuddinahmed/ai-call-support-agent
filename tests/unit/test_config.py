from support_agent.config import Settings, resolve_publishable_key


def test_defaults():
    s = Settings(_env_file=None)
    assert s.store_url == "http://localhost:9000"
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_model == "qwen3.5:4b-q4_K_M"
    assert s.llm_api_key == "ollama"
    assert s.llm_reasoning_effort == "low"
    assert s.max_tool_rounds == 6
    assert s.catalog_cache_seconds == 300
    assert s.voice_stt_model == "mlx-community/whisper-large-v3-turbo"
    assert s.voice_tts_voice == "af_heart"
    assert s.voice_asterisk_path == "/api/voice/asterisk"
    assert s.allowed_hosts == ["localhost", "127.0.0.1", "host.docker.internal"]
    assert "demo-store/docker-compose.yml" in s.store_key_file_command


def test_env_overrides_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemma")
    assert Settings(_env_file=None).llm_model == "gemma"


def test_set_key_needs_no_command():
    s = Settings(_env_file=None, store_publishable_key="pk_x", store_key_file_command="false")
    assert resolve_publishable_key(s) == "pk_x"
