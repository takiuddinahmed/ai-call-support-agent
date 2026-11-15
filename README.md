# Support agent

This project is a local AI customer support agent for the demo store Robo Parts BD. A customer can find products by need, place a cash-on-delivery order, and get the status of an order. The agent talks in Bangla, English, or a mix of both.

The agent has a text chat and English voice calls. A voice call can come from the browser or from a SIP softphone (a phone app that uses an IP network) through Asterisk. Both use the same core: the instructions, the store tools, and the store client. All AI runs locally on the Mac. The project uses no cloud AI service.

## Requirements

- macOS on Apple Silicon. The project was measured on an Apple M5 Pro with 24 GB.
- [uv](https://docs.astral.sh/uv/) for Python 3.13.
- Ollama, running on macOS (not in Docker, because Docker on macOS cannot use the Apple GPU).
- The demo store in `../demo-store`, running with `docker compose up -d`.

## Start

1. Download the default model:

   ```sh
   ollama pull qwen3.5:4b-q4_K_M
   ```

2. Copy the example configuration:

   ```sh
   cp .env.example .env
   ```

3. Start the chat app from the `agent` folder:

   ```sh
   uv run python -m support_agent
   ```

4. Open `http://localhost:8100`.

The agent reads the store publishable key from the store container. To set it yourself, put it in `STORE_PUBLISHABLE_KEY` in `.env`.

## The chat page

You can log in with a store account, or chat as a guest. The sample accounts use the password `Demo1234!`, for example `rahim@example.com`.

A logged-in customer can ask for "my orders". A guest must give the order number and the email or phone of the order.

To place an order, the agent first shows a summary with the total from the store. It places the order only after a clear yes in a later message. The code enforces this rule, so the model cannot skip it.

## The debug panel

The right side of the page shows each customer turn:

- each model call, with the time to the first token, the total time, and the tokens per second
- each tool call, with its input, its output, and its time
- the total time of the turn

Use the "Model" menu to switch between the models in Ollama. Use the "Channel" menu to switch to the voice style: short answers with no lists and with numbers in words.

## Voice calls

The voice agent speaks English. It listens with Whisper (speech-to-text), thinks with the Ollama model, and speaks with Kokoro (text-to-speech). The server must run on the branch `feat/voice-agent`.

Rules for all voice calls:

- The server takes one call at a time. A second call gets a busy answer.
- The first call after a server start waits about 50 s for the model load.
- A phone call is a guest call. The caller cannot log in by phone.
- The agent asks for a phone number, not an email. A guest order without an email gets an email made from the phone number.
- Before the agent places an order, it reads back what it heard. The caller must say a clear yes.

### Browser call

1. Start the server with `uv run python -m support_agent`.
2. Open `http://localhost:8100/voice`.
3. Log in or choose "Call as guest". Then choose "Start call".

Use headphones. Without them, the agent hears its own voice from the speakers. The page also shows a debug panel with the time of each step of each turn.

### Phone call through Asterisk

Asterisk is in the demo store, on the branch `feat/asterisk` of `../demo-store`. Follow these steps:

1. In `../demo-store/.env`, set `ASTERISK_LAN_IP`, `ASTERISK_PHONE_PASSWORD`, and `AGENT_PHONE_TOKEN`. Get the Mac LAN address with `ipconfig getifaddr en0`.
2. Set the same `AGENT_PHONE_TOKEN` in the `.env` of this folder. If the two values differ, every phone call fails. If the value is empty, the agent refuses every phone call.
3. Start Asterisk in `../demo-store`:

   ```sh
   docker compose up -d asterisk
   ```

4. Start the agent server on this branch.
5. Register a softphone with the configuration below.
6. Dial 2000 for the agent. Dial 600 for an echo test of the phone link.

If the Mac LAN address changes, change `ASTERISK_LAN_IP` and run `docker compose up -d asterisk` again.

Softphone configuration. These are the same in Linphone (iPhone, on the same Wi-Fi as the Mac) and in Telephone (the Mac app from the Mac App Store):

| Item | Value |
| --- | --- |
| Domain | The Mac LAN address with port 5060, for example `192.168.10.239:5060` |
| User | `1001` |
| Password | The value of `ASTERISK_PHONE_PASSWORD` in `../demo-store/.env` |
| Transport | UDP |
| Codecs | G.722 first, then G.711 (PCMU and PCMA). Turn off the others. |

In Linphone, choose "Use a SIP account", not the Linphone account. Allow the local network permission on iOS, and keep the app open. To check the registration, run `docker compose exec asterisk asterisk -rx "pjsip show contacts"` in `../demo-store`. Contact 1001 must show `Avail`.

The phone route is `/api/voice/asterisk`. Asterisk sends `AGENT_PHONE_TOKEN` as the HTTP Basic password. The agent hangs up a phone call after `MAX_CALL_SECONDS`.

### Voice benchmark

The voice benchmark has two parts. Each part writes a report in `reports/`.

```sh
uv run python -m support_agent.voice.bench make-audio                  # test audio in bench-audio/
uv run python -m support_agent.voice.bench stt --model mlx-community/whisper-small-mlx --rate 16000
uv run python -m support_agent.voice.bench calls --rate 16000          # full calls, about 10 minutes
```

- `stt` measures the word error rate and the time of one Whisper model. Use `--rate 8000` for the phone band.
- `calls` plays scripted caller audio through the full pipeline and measures the pause. Stop the other servers on port 8100 first, because a second call can slow a turn.

The first numbers are in `reports/2026-11-15-voice-baseline.md`. The pause median is 3.37 s on turns without tools. The slowest step is the first token of the model.

## The benchmark

The benchmark runs 26 scripted conversations in English, Bangla, and mixed language. It checks the tool calls and the facts in each answer, and it measures the speed.

```sh
uv run python -m support_agent.bench --model qwen3.5:4b-q4_K_M
```

Options:

- `--cases <glob>` runs only some case files from `src/support_agent/bench/cases/`.
- `--reasoning <none|low|medium|high|"">` changes the hidden thinking for this run.
- `--out <folder>` sets the report folder. The default is `reports/`.

The order cases place real guest orders in the demo store, with emails such as `bench-<run id>@example.com`.

Unload other models before a run, so that the numbers belong to one model:

```sh
ollama ps
ollama stop <model>
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `STORE_URL` | `http://localhost:9000` | The store API |
| `STORE_PUBLISHABLE_KEY` | empty | The store key. Empty reads it from the store container. |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | The OpenAI-compatible model server |
| `LLM_MODEL` | `qwen3.5:4b-q4_K_M` | The default model |
| `LLM_REASONING_EFFORT` | `low` | Hidden thinking: `none`, `low`, `medium`, `high`, or empty for the model default |
| `LLM_REASONING_EFFORT_AFTER_TOOLS` | empty | Hidden thinking of the model calls after a tool result in the same turn. Empty uses `LLM_REASONING_EFFORT`. |
| `MAX_TOOL_ROUNDS` | `6` | Tool rounds per turn. After the last round, the agent answers without tools. |
| `MAX_PROMPT_TOKENS` | `3200` | The prompt budget in estimated tokens. It must stay under the model context. See "Model context". |
| `CATALOG_CACHE_SECONDS` | `300` | How long the agent keeps the product list |
| `AGENT_PHONE_TOKEN` | empty | Shared secret of the Asterisk phone route. Use the same value in `../demo-store/.env`. Empty refuses every phone call. |
| `MAX_CALL_SECONDS` | `900` | The agent hangs up a phone call after this time |
| `VOICE_STT_MODEL` | `mlx-community/whisper-large-v3-turbo` | The Whisper model for speech-to-text |
| `VOICE_TTS_VOICE` | `af_heart` | The Kokoro voice of the agent |
| `VOICE_ASTERISK_PATH` | `/api/voice/asterisk` | The WebSocket path for phone calls |
| `ALLOWED_HOSTS` | `["localhost","127.0.0.1","host.docker.internal"]` | Host names that the server accepts |

The debug routes accept only clients on this computer. Docker Desktop shows each container as `127.0.0.1`, so the containers on this Mac are trusted local clients for the debug routes.

## Tests

```sh
uv run pytest -m "not store and not llm"   # unit tests, no network
uv run pytest -m store                     # needs the demo store
uv run pytest -m llm                       # needs Ollama and the store
```

## Model comparison

The full report is in `reports/2026-11-15-model-comparison.md`.

| Model | Pass | First token median | Memory after load |
| --- | --- | --- | --- |
| `qwen3.5:4b-q4_K_M` (default) | 96% | 2.1 s | 4.2 GB |
| `qwen3.5:4b` (MLX) | 96% | 1.9 s | 4.4 GB, grows to 15 GB |
| `qwen3.5:4b-mtp-q4_K_M` | 92% | 2.0 s | 2.8 GB |
| `qwen3.5:9b-q4_K_M` | 92% | 3.3 s | 4.9 GB |
| `gemma4:e4b-it-q4_K_M` | 92% | 2.2 s | 3.6 GB |

Two findings matter for speed:

- With the model default, `qwen3.5:4b` thinks for about 18 seconds before the first word. `LLM_REASONING_EFFORT=low` brings this to about 2 seconds.
- With `none`, the first word comes in about 0.2 seconds, but the model skips the store tools and invents prices. The next work is to get this speed without that error.

## Model context

Ollama runs `qwen3.5:4b-q4_K_M` with a context of 4096 tokens. The system prompt and the tool specs use about 1850 tokens. If the prompt fills the context, no room stays for the hidden thinking, and the model gives an empty answer.

The agent keeps each prompt under `MAX_PROMPT_TOKENS` (estimated tokens). The estimate is ASCII characters / 3 plus other characters / 1.5. It is a little too high, so the real prompt is smaller. If the prompt is too large, the agent first replaces old tool results with a short note. Then it drops the oldest whole customer turns. A tool call and its result stay together. The current turn and the pending order summary always stay. `session.messages` keeps the full history.

If a model call still gives no text and no tool call, the agent says "Sorry, I did not get that. Can you say it again?". The debug panel shows an error row.

To give the model more history, you can use a larger context:

1. Make a model variant with a larger context. Ollama ignores `num_ctx` in a request to the OpenAI-compatible API.

   ```sh
   printf 'FROM qwen3.5:4b-q4_K_M\nPARAMETER num_ctx 8192\n' > /tmp/Modelfile.ctx8k
   ollama create qwen3.5:4b-q4_K_M-ctx8k -f /tmp/Modelfile.ctx8k
   ```

   As an alternative, set `OLLAMA_CONTEXT_LENGTH=8192` for the Ollama server and start it again. This changes all models.
2. Set `LLM_MODEL=qwen3.5:4b-q4_K_M-ctx8k` and `MAX_PROMPT_TOKENS=6400` in `.env`.

The 8192 context measured 3.4 GB in `ollama ps`, and 4096 measured 3.2 GB. The details are in `reports/2026-11-15-task-11-context-and-effort.md`.

## Project layout

| Path | Content |
| --- | --- |
| `src/support_agent/config.py` | Configuration |
| `src/support_agent/store/` | The store client and the store data types |
| `src/support_agent/specs.py` | Product search by words and spec values |
| `src/support_agent/tools.py` | The store tools and their rules |
| `src/support_agent/instructions.py` | The system prompt, chat and voice styles |
| `src/support_agent/loop.py` | One customer turn, with timed events |
| `src/support_agent/llm.py` | The client for the model server |
| `src/support_agent/server/` | The chat server, the chat page, and the voice page |
| `src/support_agent/voice/` | The voice pipeline, the browser and phone routes, the debug hub, and the voice benchmark |
| `src/support_agent/bench/` | The benchmark and its cases |
| `reports/` | Benchmark reports |
