# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local AI customer support agent for the demo store "Robo Parts BD" (embedded and robotics parts, Bangladesh, BDT, cash on delivery). Customers find products by need, place orders, and track orders, in Bangla, English, or mixed text. This repository has the text chat and English voice calls (Pipecat, local speech models). A call comes from the browser at `/voice` or from an Asterisk IP phone.

All AI runs locally. Do not add cloud AI services. The models run in Ollama directly on macOS, because Docker on macOS cannot use the Apple GPU.

The store is a separate repository in `../demo-store` (Medusa v2 in Docker). The agent talks to it only through the store HTTP API at `http://localhost:9000`. Never read the store database from the agent.

## Commands

Run every command from this folder. The default publishable key lookup runs `docker compose -f ../demo-store/docker-compose.yml ...`, so the relative path must resolve.

```sh
uv run python -m support_agent                 # chat app at http://localhost:8100
uv run pytest -m "not store and not llm"       # unit tests, no network
uv run pytest -m store                         # needs the demo store running
uv run pytest -m llm                           # needs Ollama and the store
uv run pytest tests/unit/test_tools.py::test_tool_specs_names   # one test
uv run ruff check src tests
uv run python -m support_agent.bench --model qwen3.5:4b-q4_K_M  # benchmark, report in reports/
uv run python -m support_agent.voice.bench make-audio          # test audio in bench-audio/
uv run python -m support_agent.voice.bench stt --model mlx-community/whisper-small-mlx --rate 16000
uv run python -m support_agent.voice.bench calls --rate 16000   # full call benchmark, about 10 min
uv run python -m support_agent.voice.bench report <json>        # rebuild a calls report from its JSON
uv run python -m support_agent.bench --model <m> --cases 'src/support_agent/bench/cases/1-*.yaml' --reasoning none
```

Start the store with `docker compose up -d` in `../demo-store`. Its compose file pins `name: ai-audio-call`, so the data volumes keep their names.

The `store` tests and the benchmark order cases place real orders in the demo store. Store tests register a new customer for each run. Do not place test orders as the sample customers (for example `rahim@example.com`), because the demo shows their order lists.

## Architecture

The package `src/support_agent` is a core library plus the chat server and the voice package. The core has no web code, so both front ends import it.

- `store/client.py` is the only code that calls the store API. It caches the product list (`catalog_cache_seconds`). `quote()` makes a temporary store cart to read the delivery fee. `place_order()` runs the full Medusa cart flow. Delivery zone rule: city "Dhaka" (trimmed, any case) is "Inside Dhaka", other cities are "Outside Dhaka". The store enforces the same rule.
- `specs.py` searches products by text, category, and typed spec values from `metadata.specs`. `SPEC_KEYS` copies the key list from the demo-store README. Text search matches whole words, drops stop words, prefers products that contain all words, and falls back to any word.
- `tools.py` defines the tools that the model can call (`TOOL_SPECS`) and `ToolRunner`. `ToolRunner.run` never raises: every failure becomes `ToolResult(ok=False, content=<message for the model>)`. Keep it that way, because a raised error breaks the streamed turn.
- `session.py` holds the conversation, the identity (logged-in customer or guest), and the pending order summaries. On each login and logout, `start_over()` clears everything.
- `instructions.py` builds the system prompt for the channel `chat` or `voice`. Voice style means one to three sentences, no lists, numbers in words.
- `loop.py` `run_turn()` streams one customer turn and yields timed `AgentEvent`s. The event types are `turn_start`, `llm_start`, `first_token`, `text`, `tool_call_start`, `llm_end`, `tool_start`, `tool_end`, `answer`, and `error`. When the model starts to name a tool, `tool_call_start` comes before the arguments end. The voice filler uses it. If the caller does not close the generator, `run_turn` is not cancel-safe. Always use `contextlib.aclosing(run_turn(...))`. After `max_tool_rounds`, it makes one last model call with no tools, so that the model must answer.
- `llm.py` is the OpenAI-compatible streaming client. It sends `reasoning_effort` from the configuration.
- `server/app.py` serves the chat page and streams the events as server-sent events to the debug panel. `create_app(settings, store=, llm=, models=)` takes injected parts for tests. Sessions live in memory.
- `voice/` is the voice package. It imports Pipecat only inside functions, so the chat server loads no Pipecat module.
  - `pipeline.py` builds the call pipeline: VAD, smart turn, Whisper, `TurnTextGate`, `AgentProcessor`, Kokoro, and `SpokenTextTracker`.
  - `agent_processor.py` runs `run_turn` for each heard text, speaks a filler at `tool_call_start`, and records the spoken text.
  - `routes.py` has `/voice`, `POST /api/voice/offer` (browser WebRTC), the SSE and call routes of the debug panel, and the phone WebSocket at `voice_asterisk_path`.
  - `asterisk.py` has the Asterisk control parser and the frame serializer.
  - `debug_hub.py`, `timing.py`, and `deps.py` have the call events, the step timer, and the shared Whisper and Kokoro models.
  - `bench/` has the Whisper benchmark (`stt.py`, `wer.py`) and the call benchmark (`pipeline_bench.py`, `calls.yaml`).
  - One call runs at a time, for the browser and the phone together.
- `bench/runner.py` runs the YAML conversations in `bench/cases/` and checks tool calls, answer facts, and order placement. `{live:price:<handle>}` in an expectation becomes the live store price, and `{bench_email}` in a message becomes a run-specific guest email.

## Rules that the code enforces

These rules come from the design in `docs/superpowers/specs/2026-11-15-chat-agent-design.md`. Do not weaken them.

- Order approval: `prepare_order` stores one summary (it clears older ones) with `customer_turn_at_creation`. `place_order` accepts only that summary ID, and only after a later customer message. It removes the summary before the store call, so a retry cannot place a second order. The model judges whether the customer's reply means "yes".
- Privacy: a guest sees an order only with the order number and the matching email or phone. A tracking number alone gives the delivery status and history, never the items or the address.
- Facts: prices, stock, products, and order status come only from tool results. The instructions forbid a model-made order summary or total.

## Voice rules

- Before `place_order`, the voice agent reads back what it heard (name, address, city, phone) and the caller must say yes. This guards against speech-to-text errors.
- The voice agent answers in English only.
- A voice guest gives a phone number, not an email. If the email is empty, `prepare_order` makes one with `voice_guest_email(phone)`, in the form `voice-880XXXXXXXXXX@guest.robopartsbd.local`.
- The phone route makes a guest session that no HTTP route can reach. A phone caller cannot log in.

## Security settings

- `allowed_hosts` (env `ALLOWED_HOSTS`): the Host headers that the server accepts. The default includes `host.docker.internal`, so Asterisk in Docker can reach the phone route. Other HTTP routes refuse that host with 403.
- `phone_token` (env `AGENT_PHONE_TOKEN`): the shared secret of the phone route. Asterisk sends it as the HTTP Basic password. Empty refuses every phone call. The value must match `../demo-store/.env`.
- The phone route refuses a request with an `Origin` header, so a web page cannot start a phone call.
- The debug routes accept only local clients. Docker Desktop shows each container as `127.0.0.1`, so containers on this Mac are trusted clients.
- `max_call_seconds` ends a phone call after 900 s.

## Voice measurements

The starting numbers are in `reports/2026-11-15-voice-baseline.md`. The pause median is 3.37 s (goal 1.5 s). In one browser call, the first model token took about 70 percent of the pause. The default Whisper model is `mlx-community/whisper-small-mlx`. Known problems follow. The 4096-token context fills in a long call. The turn-end model splits a turn at a pause inside a sentence. Whisper can mishear city names.

## Tests

`tests/unit/fakes.py` has `FakeStore` (same method names as `StoreClient`) and `FakeLlm` (plays scripted rounds: a string is a text answer, a list of `(tool_name, json_arguments)` is a tool round). Use them for loop, tool, server, and benchmark tests. A change to a `StoreClient` method signature needs the same change in `FakeStore`.

## Model findings

The measurements are in `reports/2026-11-15-model-comparison.md`. Two of them shape the defaults:

- `LLM_REASONING_EFFORT=low` is the default. The model default thinking took about 18 s to the first token. `none` took about 0.2 s, but `qwen3.5:4b` then skipped the tools and invented prices.
- The default model is `qwen3.5:4b-q4_K_M` (llama.cpp runner). The MLX build `qwen3.5:4b` grew to about 15 GB during a benchmark run. The memory budget for all models is 12 GB.

The benchmark memory columns are not reliable. The Ollama estimate adds up all loaded models, and the macOS footprint misses the mapped weights of the llama.cpp runner. Stop other models (`ollama ps`, `ollama stop <model>`) before a benchmark run.
