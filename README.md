# Support agent

This project is a local AI customer support agent for the demo store Robo Parts BD. A customer can find products by need, place a cash-on-delivery order, and get the status of an order. The agent talks in Bangla, English, or a mix of both.

This is the text chat phase. The voice phase will use the same core: the instructions, the store tools, and the store client. All AI runs locally on the Mac. The project uses no cloud AI service.

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
| `MAX_TOOL_ROUNDS` | `6` | Tool rounds per turn. After the last round, the agent answers without tools. |
| `CATALOG_CACHE_SECONDS` | `300` | How long the agent keeps the product list |
| `AGENT_PHONE_TOKEN` | empty | Shared secret of the Asterisk phone route. Use the same value in `../demo-store/.env`. Empty refuses every phone call. |
| `MAX_CALL_SECONDS` | `900` | The agent hangs up a phone call after this time |
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
| `src/support_agent/server/` | The chat server and the chat page |
| `src/support_agent/bench/` | The benchmark and its cases |
| `reports/` | Benchmark reports |
