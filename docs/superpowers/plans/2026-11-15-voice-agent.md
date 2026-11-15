# Voice Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** English voice calls with the support agent from the browser and from Linphone on an iPhone. A debug panel shows the time of each step.

**Architecture:** A Pipecat pipeline per call: transport, speech detection and turn end, MLX Whisper, an agent processor that runs the existing `run_turn()`, Kokoro, transport. The browser call uses the Pipecat WebRTC transport. The phone call goes from Linphone to Asterisk in Docker, and Asterisk sends the audio over a WebSocket to the agent server. A debug hub collects the timed events of all calls and streams them to the voice page.

**Tech Stack:** Python 3.13, uv, FastAPI, Pipecat 1.16.0, Ollama, Asterisk 22 with `chan_websocket`, Docker Compose. The Pipecat extras are `webrtc`, `websocket`, `silero`, `local-smart-turn`, `mlx-whisper`, and `kokoro`.

**Spec:** `docs/superpowers/specs/2026-11-15-voice-agent-design.md`

## Global Constraints

- All AI runs locally. No cloud AI service. Models run on macOS, not in Docker.
- All AI models together stay under 12 GB.
- English only on the voice channel. The chat channel keeps Bangla.
- One call at a time.
- Correctness always wins over speed. A speed fix that makes the model skip a tool or invent a fact is rejected.
- The voice code imports the core. It does not copy it. Small core changes are allowed. The chat rules must not get weaker, and every existing test must pass.
- `ToolRunner.run` never raises.
- No change to the demo store code. The Asterisk service is a new folder and a new compose service in `demo-store`.
- Run all agent commands from `agent/`. All English text follows the ASD-STE100 rules in the global CLAUDE.md.
- Pause target: a median of 1.5 s or less on turns without a tool call. On tool turns, the filler phrase starts within 1.5 s. These are goals for Tasks 11 and 12, not gates.

## Review Focus

1. The caller interrupts the order summary, then says "yes". The agent must not place the order. It reads the summary again. Test: Task 3 `test_interrupted_summary_is_not_heard`.
2. An interruption arrives while `place_order` runs. The order is placed once and the history records it. Test: Task 1 `test_cancel_during_tool_keeps_result`.
3. The caller says "hello?" during a long store search. The turn must finish, and "hello?" becomes the next turn. Test: Task 3 `test_speech_before_audio_is_held`.
4. A phone caller hangs up in the middle of a turn. The server must close the call without an error loop and free the session. Test: Task 7 `test_hangup_ends_call`.
5. Whisper hears nothing or only noise (an empty transcription). The agent must not start a model call. Test: Task 3 `test_empty_transcription_is_ignored`.

---

### Task 1: Core loop: early tool event and cancel-safe history

**Files:**
- Modify: `src/support_agent/loop.py`
- Modify: `src/support_agent/llm.py`
- Modify: `tests/unit/fakes.py`
- Test: `tests/unit/test_loop.py`

**Interfaces:**
- Produces: a new event type `"tool_call_start"` with data `{"name": str, "round": int}`. `run_turn()` yields it once per tool call, on the first streamed piece of that call, before `llm_end`.
- Produces: `run_turn()` is cancel-safe. On `asyncio.CancelledError`, it leaves `session.messages` valid and then re-raises.
- Produces: `LlmClient.stream(model, messages, tools, reasoning_effort: str | None = None)`. `None` uses the configuration value. Task 12 uses this.
- Produces: `FakeLlm(rounds, error=None, delay_s: float = 0.0)`. `delay_s` waits before each chunk, so tests can cancel during a stream. `FakeLlm.efforts_seen: list[str | None]` records the `reasoning_effort` of each call.

- [ ] **Step 1: Write the failing tests**

```python
async def test_tool_call_start_comes_before_llm_end():
    events = [e async for e in run_turn(session, "price of esp32", FakeLlm([[("search_products", '{"text":"esp32"}')], "It costs 550 taka."]), runner, "m", 6)]
    kinds = [e.type for e in events]
    assert kinds.index("tool_call_start") < kinds.index("llm_end")
    assert events[kinds.index("tool_call_start")].data == {"name": "search_products", "round": 1}

async def test_cancel_during_stream_drops_partial_answer():
    # FakeLlm(["one two three four"], delay_s=0.05). Cancel the consumer task after the first "text" event.
    # Expect: session.messages == [{"role": "user", "content": "hi"}]

async def test_cancel_during_tool_keeps_result():
    # FakeStore.place_order waits 0.1 s. Cancel during the place_order tool call.
    # Expect: the cancel waits for the tool. store.placed has 1 order. The last message is the
    # "tool" message with the order. Every assistant tool_call id has one tool message.

async def test_cancel_between_tools_fills_missing_results():
    # A round with two tool calls. Cancel during the first tool.
    # Expect: the second call gets a tool message with content
    # "Not run: the customer interrupted." and the store never sees the second call.
```

- [ ] **Step 2: Run the tests and make sure that they fail**

Run: `uv run pytest tests/unit/test_loop.py -v`
Expected: the four new tests FAIL.

- [ ] **Step 3: Implement the changes**

In `run_turn()`: if a delta brings a new tool call index with a name, yield `tool_call_start`. Run each tool inside `asyncio.shield()`. On cancel, await the shielded task and append its result. Then add the "Not run" message for the calls that did not run, and re-raise. On cancel during a model stream, drop the partial text and keep the earlier messages. In `llm.py`, add the `reasoning_effort` keyword. Add `delay_s` and `efforts_seen` to `FakeLlm`.

- [ ] **Step 4: Run all unit tests**

Run: `uv run pytest -m "not store and not llm"`
Expected: all pass, the 70 earlier tests too.

- [ ] **Step 5: Commit**

```bash
git add src/support_agent/loop.py src/support_agent/llm.py tests/unit/fakes.py tests/unit/test_loop.py
git commit -m "feat: add an early tool event and cancel-safe history to the agent loop"
```

### Task 2: Core voice rules: instructions, heard summaries, phone email

**Files:**
- Modify: `src/support_agent/instructions.py`
- Modify: `src/support_agent/session.py`
- Modify: `src/support_agent/tools.py`
- Test: `tests/unit/test_instructions.py`, `tests/unit/test_tools.py`

**Interfaces:**
- Produces: `OrderSummary.heard: bool = False`.
- Produces: `Session.mark_summaries_heard(created_at_turn: int) -> None`. It sets `heard=True` on the summaries with `customer_turn_at_creation == created_at_turn`.
- Produces: `voice_guest_email(phone: str) -> str` in `tools.py`. It returns `voice-8801XXXXXXXXX@guest.robopartsbd.local` with the normalized digits.

- [ ] **Step 1: Write the failing tests**

```python
def test_voice_instructions_force_english():
    text = build_instructions("voice", Identity.guest(), NOW)
    assert "Answer in English" in text
    assert "language of the customer's last message" not in text
    assert "digit by digit" in text
    assert "phone number" in text  # guests give the phone number for order status

def test_chat_instructions_keep_language_rule():
    assert "language of the customer's last message" in build_instructions("chat", Identity.guest(), NOW)

async def test_voice_place_order_needs_heard_summary():
    # channel "voice": prepare_order, next customer turn, place_order -> ok=False,
    # content contains "Read the full summary to the customer again". store.placed == [].
    # After session.mark_summaries_heard(turn): place_order -> ok=True.

async def test_chat_place_order_needs_no_heard_flag():
    # channel "chat": the old flow still places the order.

async def test_voice_guest_order_without_email_uses_phone_email():
    # channel "voice", guest, prepare_order with address phone "01712345678" and no email
    # -> result["email"] == "voice-8801712345678@guest.robopartsbd.local"

async def test_chat_guest_order_without_email_still_asks_for_email():
    # channel "chat" -> ok=False, content contains "email".
```

- [ ] **Step 2: Run the tests and make sure that they fail**

Run: `uv run pytest tests/unit/test_instructions.py tests/unit/test_tools.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement the changes**

Move the language rule out of `BASE` into the two styles. The voice style says: "Answer in English. The customer can use another language, but you answer in English." It tells the agent to say order numbers digit by digit. It tells the agent to ask a guest for the phone number of the order, not the email. In `_tool_place_order`, check the voice channel first. If `session.channel == "voice"` and the summary is not heard, raise `ToolError("Read the full summary to the customer again, then ask for a clear yes.")`. Keep the summary in that case. In `_order_contact`, on the voice channel with no email, use `voice_guest_email(fields["phone"])` after the phone check.

- [ ] **Step 4: Run all unit tests**

Run: `uv run pytest -m "not store and not llm"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/support_agent tests/unit
git commit -m "feat: add the voice rules for language, heard summaries, and guest email"
```

### Task 3: Voice dependencies and the agent processor

**Files:**
- Modify: `pyproject.toml`, `src/support_agent/config.py`, `.env.example`
- Create: `src/support_agent/voice/__init__.py`
- Create: `src/support_agent/voice/agent_processor.py`
- Test: `tests/unit/test_agent_processor.py`

**Interfaces:**
- Consumes: `run_turn()` and its events from Task 1. `Session.mark_summaries_heard()` from Task 2.
- Produces: `AgentProcessor(session: Session, llm: LlmClient, runner: ToolRunner, model: str, max_tool_rounds: int, on_event: Callable[[str, dict], Awaitable[None]])`, a Pipecat `FrameProcessor`.
  - Input: `TranscriptionFrame`, `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`, `InterruptionFrame`.
  - Output: `LLMFullResponseStartFrame`, `LLMTextFrame`, `LLMFullResponseEndFrame`, `TTSSpeakFrame` for the filler.
  - `on_event(kind, data)` receives every `AgentEvent` of the turn (kind = event type, data = event data plus `t_ms`), and also `"heard"` and `"filler"`.
- Produces: `FILLER_PHRASES: list[str]` in `agent_processor.py`.
- Produces: new configuration values `voice_stt_model: str = "mlx-community/whisper-large-v3-turbo"`, `voice_tts_voice: str = "af_heart"`, `voice_asterisk_path: str = "/api/voice/asterisk"`.

- [ ] **Step 1: Add the dependencies**

Run: `uv add "pipecat-ai[webrtc,websocket,silero,local-smart-turn,mlx-whisper,kokoro]==1.16.0"`
Expected: `uv run python -c "import pipecat; print(pipecat.__version__)"` prints `1.16.0`.

- [ ] **Step 2: Write the failing tests**

Use `pipecat.tests.utils.run_test` with `FakeLlm` and `FakeStore`.

```python
async def test_transcription_streams_answer_frames():
    # TranscriptionFrame("hi") -> down frames: LLMFullResponseStartFrame, LLMTextFrame..., LLMFullResponseEndFrame

async def test_filler_spoken_on_tool_call():
    # FakeLlm([[("search_products", '{"text":"esp32"}')], "It costs 550 taka."])
    # -> the first TTSSpeakFrame comes before any LLMTextFrame, and its text is in FILLER_PHRASES

async def test_no_filler_after_answer_text():
    # A round with text "Sure." and a tool call -> no TTSSpeakFrame

async def test_empty_transcription_is_ignored():
    # TranscriptionFrame("  ") -> no model call (FakeLlm.seen == [])

async def test_speech_before_audio_is_held():
    # FakeLlm with delay_s. Send TranscriptionFrame("price of esp32"), then
    # InterruptionFrame and TranscriptionFrame("hello?") before BotStartedSpeakingFrame.
    # -> the first turn finishes. Then a second turn runs with "hello?". session.customer_turns == 2.

async def test_interruption_while_speaking_cancels_turn():
    # TranscriptionFrame, BotStartedSpeakingFrame, InterruptionFrame
    # -> the turn task is cancelled. session.messages is valid (Task 1 rules).

async def test_heard_after_full_answer():
    # A turn with prepare_order, then BotStoppedSpeakingFrame with no interruption
    # -> the summary has heard=True and on_event received "heard".

async def test_interrupted_summary_is_not_heard():
    # The same, but InterruptionFrame before BotStoppedSpeakingFrame -> heard stays False.
    # The next turn "yes" -> place_order returns ok=False and store.placed == [].
```

- [ ] **Step 3: Run the tests and make sure that they fail**

Run: `uv run pytest tests/unit/test_agent_processor.py -v`
Expected: FAIL with an import error for `support_agent.voice.agent_processor`.

- [ ] **Step 4: Implement `AgentProcessor`**

Run each turn as an asyncio task. Keep a queue of held transcriptions. Track `bot_spoke_this_turn`. If `bot_spoke_this_turn` is true, cancel the turn on `InterruptionFrame`. If it is false, do not pass the `InterruptionFrame` on, and hold the next transcription. If no text frame went out in this turn, push the filler on the first `tool_call_start`. Mark the summaries heard on the `BotStoppedSpeakingFrame` that ends the last answer of a turn with no interruption after its start.

- [ ] **Step 5: Run all unit tests**

Run: `uv run pytest -m "not store and not llm"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .env.example src/support_agent/config.py src/support_agent/voice tests/unit/test_agent_processor.py
git commit -m "feat: add the voice agent processor with filler and interruption rules"
```

### Task 4: Call timing and the debug hub

**Files:**
- Create: `src/support_agent/voice/timing.py`
- Create: `src/support_agent/voice/debug_hub.py`
- Test: `tests/unit/test_timing.py`, `tests/unit/test_debug_hub.py`

**Interfaces:**
- Produces: `TurnTimer(on_event)`, a Pipecat `BaseObserver`. It sends these events with `ms` values through `on_event`:
  - `"speech_end"`: from `VADUserStoppedSpeakingFrame` to `UserStoppedSpeakingFrame` (the turn-end decision).
  - `"stt"`: the Whisper time, with the heard `text`.
  - `"first_audio"`: from the turn-end decision to the first `TTSAudioRawFrame` sent out.
  - `"pause"`: from `VADUserStoppedSpeakingFrame` to the first `TTSAudioRawFrame` sent out.
- Produces: `DebugHub` with `start_call(channel: Literal["browser", "phone"], caller: str) -> str` (a call ID), `publish(call_id: str, kind: str, data: dict) -> None`, `end_call(call_id: str) -> None`, `subscribe() -> AsyncIterator[dict]`, and `calls() -> list[dict]`. It keeps the last 20 calls in memory.

- [ ] **Step 1: Write the failing tests**

```python
async def test_pause_measures_speech_end_to_first_audio():
    # Feed the observer frames with fake clock times 0.0 (VAD stop), 0.3 (turn end),
    # 0.6 (transcription), 1.4 (first TTS audio) -> pause == 1400, first_audio == 1100, speech_end == 300

async def test_hub_streams_events_to_subscribers():
    # subscribe, start_call("phone", "1001"), publish(...) -> the subscriber gets call_start and the event

def test_hub_keeps_last_20_calls():
    # 25 calls -> len(calls()) == 20
```

- [ ] **Step 2: Run the tests and make sure that they fail**

Run: `uv run pytest tests/unit/test_timing.py tests/unit/test_debug_hub.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement `TurnTimer` and `DebugHub`**

`TurnTimer` takes a `clock: Callable[[], float] = time.perf_counter` argument, so the test can set the times.

- [ ] **Step 4: Run all unit tests**

Run: `uv run pytest -m "not store and not llm"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/support_agent/voice tests/unit/test_timing.py tests/unit/test_debug_hub.py
git commit -m "feat: add call timing and a debug hub for voice calls"
```

### Task 5: Browser call: pipeline, routes, and the voice page

**Files:**
- Create: `src/support_agent/voice/pipeline.py`
- Create: `src/support_agent/voice/routes.py`
- Create: `src/support_agent/server/static/voice.html`
- Modify: `src/support_agent/server/app.py`
- Test: `tests/unit/test_voice_routes.py`

**Interfaces:**
- Consumes: `AgentProcessor` (Task 3), `TurnTimer` and `DebugHub` (Task 4).
- Produces: `build_task(transport: BaseTransport, session: Session, deps: VoiceDeps, hub: DebugHub, call_id: str, audio_rate: int) -> PipelineTask`. The pipeline order is:
  1. `transport.input()`
  2. `UserTurnProcessor`, with a Silero VAD start strategy and a `LocalSmartTurnAnalyzerV3` stop strategy
  3. `WhisperSTTServiceMLX`, with language English and model `settings.voice_stt_model`
  4. `AgentProcessor`
  5. `KokoroTTSService`, with voice `settings.voice_tts_voice`
  6. `transport.output()`
- Produces: `VoiceDeps(settings, store, llm)`, and a `voice_router(deps, sessions, hub) -> APIRouter` with:
  - `GET /voice` serves `voice.html`.
  - `POST /api/voice/offer` takes `{session_id, sdp, type}` and returns the WebRTC answer from `SmallWebRTCRequestHandler`. It sets the session channel to `"voice"`.
  - `GET /api/voice/events` streams the `DebugHub` events as server-sent events.
  - `GET /api/voice/calls` returns `DebugHub.calls()`.
- Produces: `create_app()` includes the voice router. `/api/session`, `/api/login`, and `/api/logout` work for the voice page too.

- [ ] **Step 1: Write the failing tests**

```python
def test_voice_page_is_served(client):
    assert client.get("/voice").status_code == 200

def test_offer_needs_known_session(client):
    assert client.post("/api/voice/offer", json={"session_id": "nope", "sdp": "x", "type": "offer"}).status_code == 404

def test_calls_list_starts_empty(client):
    assert client.get("/api/voice/calls").json() == []
```

- [ ] **Step 2: Run the tests and make sure that they fail**

Run: `uv run pytest tests/unit/test_voice_routes.py -v`
Expected: FAIL with 404 on `/voice`.

- [ ] **Step 3: Implement the pipeline, the routes, and the page**

Check the `UserTurnProcessor` and strategy parameters in the installed Pipecat source (`pipecat/turns/`). Load the models at the first call, not at server start. `voice.html` has the login form of the chat page, a "Start call" and an "End call" button, the WebRTC client code, and a debug panel. The panel lists the calls and, for each turn, the heard text, the spoken text, and the columns of "Timing" in the spec. Copy the style tokens of `index.html`, including the dark mode.

- [ ] **Step 4: Run all unit tests**

Run: `uv run pytest -m "not store and not llm"`
Expected: all pass.

- [ ] **Step 5: Make a real browser call**

Start Ollama, the store, and `uv run python -m support_agent`. Open `http://localhost:8100/voice` in Chrome, log in as `rahim@example.com`, and start a call. Ask "What is the price of the ESP32 DevKit?" and then "What are my orders?".
Expected: the agent says a filler phrase, then the price from the store. The debug panel shows all timing columns. Note the pause numbers in the task report.

- [ ] **Step 6: Commit**

```bash
git add src/support_agent tests/unit/test_voice_routes.py
git commit -m "feat: add browser voice calls with a debug panel"
```

### Task 6: Asterisk in the demo store compose project

**Files (repository `../demo-store`):**
- Create: `asterisk/Dockerfile`. If a maintained image has Asterisk 22.5 or later with `chan_websocket`, skip it.
- Create: `asterisk/pjsip.conf`, `asterisk/extensions.conf`, `asterisk/websocket_client.conf`, `asterisk/rtp.conf`, `asterisk/modules.conf`
- Modify: `docker-compose.yml`, `.env.example`, `README.md`

**Interfaces:**
- Produces: SIP account `1001` with password from `ASTERISK_PHONE_PASSWORD` (default `demo1001`) for the iPhone.
- Produces: extension `2000` dials `WebSocket/agent/c(slin16)`. The connection `agent` in `websocket_client.conf` points to `ws://host.docker.internal:8100/api/voice/asterisk`.
- Produces: codecs for `1001` in this order: `g722`, `ulaw`, `alaw`.
- Produces: published ports `5060/udp` and `10000-10049/udp`. `rtp.conf` uses the same range. `external_media_address` and `external_signaling_address` come from `ASTERISK_LAN_IP`.

- [ ] **Step 1: Pick the image**

Run: `docker run --rm <image> asterisk -V` and `docker run --rm <image> ls /usr/lib/asterisk/modules | grep chan_websocket`
Expected: version 22.5 or later, and `chan_websocket.so` exists. Write the image and the version in `asterisk/README` text inside `README.md`.

- [ ] **Step 2: Add the configuration and the compose service**

The service name is `asterisk`. Add `extra_hosts: ["host.docker.internal:host-gateway"]`.

- [ ] **Step 3: Make sure that Asterisk starts and loads the modules**

Run: `docker compose up -d asterisk && docker compose exec asterisk asterisk -rx "module show like websocket"` and `asterisk -rx "pjsip show endpoints"`
Expected: `chan_websocket.so` is running, and endpoint `1001` shows.

- [ ] **Step 4: Register Linphone**

Ask the user to set up Linphone on the iPhone with the server `<Mac LAN IP>:5060`, user `1001`, password from `.env`, transport UDP. Run `asterisk -rx "pjsip show contacts"`.
Expected: a contact for `1001` shows as `Avail`.

- [ ] **Step 5: Commit (in `../demo-store`)**

```bash
git add asterisk docker-compose.yml .env.example README.md
git commit -m "feat: add an Asterisk server for IP phone calls to the agent"
```

### Task 7: Phone transport for Asterisk

**Files:**
- Create: `src/support_agent/voice/asterisk.py`
- Modify: `src/support_agent/voice/routes.py`
- Test: `tests/unit/test_asterisk.py`

**Interfaces:**
- Produces: `AsteriskFrameSerializer(FrameSerializer)` for `chan_websocket`:
  - `deserialize(bytes)` returns `InputAudioRawFrame` (16-bit PCM, 16000 Hz, mono).
  - `deserialize(str)` reads the control messages. `MEDIA_START` gives the format and the frame size. `HANGUP` returns `EndFrame`.
  - `serialize(OutputAudioRawFrame)` returns the PCM bytes at 16000 Hz. `serialize(InterruptionFrame)` returns the text command `FLUSH_MEDIA`. `serialize(EndFrame)` returns `HANGUP`.
- Produces: the WebSocket route at `settings.voice_asterisk_path`. Each connection gets these parts:
  1. A new guest `Session` with channel `"voice"`.
  2. A `DebugHub` call with channel `"phone"`. If `MEDIA_START` has the caller number, the call gets it.
  3. A `FastAPIWebsocketTransport` with the serializer.
- Produces: one call at a time. If a call is already active, the route closes the new connection with code 1013.

- [ ] **Step 1: Write the failing tests**

```python
async def test_binary_audio_becomes_input_frame():
    frame = await AsteriskFrameSerializer().deserialize(b"\x00\x01" * 320)
    assert isinstance(frame, InputAudioRawFrame) and frame.sample_rate == 16000 and frame.num_channels == 1

async def test_interruption_flushes_media():
    assert await AsteriskFrameSerializer().serialize(InterruptionFrame()) == "FLUSH_MEDIA"

async def test_hangup_ends_call():
    assert isinstance(await AsteriskFrameSerializer().deserialize("HANGUP"), EndFrame)

def test_second_call_is_rejected(client):
    # Open one WebSocket to the route, then a second -> the second closes with code 1013
```

Check the exact control message format (plain text or JSON) against the `chan_websocket` documentation of the chosen Asterisk version, and make the tests match it.

- [ ] **Step 2: Run the tests and make sure that they fail**

Run: `uv run pytest tests/unit/test_asterisk.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement the serializer and the route**

Also send the answer audio in frames of the `optimal_frame_size` from `MEDIA_START`, with `fixed_audio_packet_size` of the transport.

- [ ] **Step 4: Run all unit tests**

Run: `uv run pytest -m "not store and not llm"`
Expected: all pass.

- [ ] **Step 5: Make a real phone call**

Start the agent server and Asterisk. Ask the user to call `2000` from Linphone and ask for the price of the ESP32 DevKit. Then the user hangs up in the middle of an answer.
Expected: the agent answers with a price from the store. The voice page lists the call as "phone" with its times. After the hangup, the server log shows the call end and no error loop.

- [ ] **Step 6: Commit**

```bash
git add src/support_agent/voice tests/unit/test_asterisk.py
git commit -m "feat: add IP phone calls through Asterisk"
```

### Task 8: Speech-to-text benchmark

**Files:**
- Create: `src/support_agent/voice/bench/__init__.py`, `src/support_agent/voice/bench/__main__.py`
- Create: `src/support_agent/voice/bench/wer.py`
- Create: `src/support_agent/voice/bench/make_audio.py`
- Create: `src/support_agent/voice/bench/sentences.yaml`
- Create: `src/support_agent/voice/bench/stt.py`
- Test: `tests/unit/test_wer.py`

**Interfaces:**
- Produces: `word_error_rate(reference: str, heard: str) -> float`. It lowercases, removes punctuation, and counts substitutions, deletions, and insertions over the reference word count.
- Produces: `sentences.yaml` with at least 30 sentences: product names (ESP32, Arduino Uno, L298N, Raspberry Pi 5), order numbers, Bangladesh phone numbers, addresses in Dhaka, and yes and no answers.
- Produces: `uv run python -m support_agent.voice.bench make-audio` writes WAV files to `bench-audio/` (in `.gitignore`), with two Kokoro voices and two macOS `say` voices, at 16000 Hz and at 8000 Hz.
- Produces: `uv run python -m support_agent.voice.bench stt --model <mlx model> [--rate 16000|8000]` writes `reports/<date>-stt-<model>-<rate>.md`. The report has these parts:
  1. The mean word error rate, and the error rate for each sentence group.
  2. The median and p90 time.
  3. A note that synthetic voices give lower error rates than real callers.

- [ ] **Step 1: Write the failing tests**

```python
def test_wer_exact_match():
    assert word_error_rate("price of the ESP32", "Price of the ESP32.") == 0.0

def test_wer_one_substitution_in_four_words():
    assert word_error_rate("price of the ESP32", "price of the ESP 32") == 0.5  # 1 sub + 1 insert over 4

def test_wer_empty_heard():
    assert word_error_rate("yes please", "") == 1.0
```

- [ ] **Step 2: Run the tests and make sure that they fail**

Run: `uv run pytest tests/unit/test_wer.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement the benchmark**

8 kHz audio is resampled to 16 kHz before Whisper, the same as in the phone pipeline.

- [ ] **Step 4: Run the tests, then run the benchmark for three models**

Run: `uv run pytest tests/unit/test_wer.py && uv run python -m support_agent.voice.bench make-audio` and then `stt` for `mlx-community/whisper-large-v3-turbo`, `mlx-community/distil-whisper-large-v3`, and `mlx-community/whisper-small-mlx`, each at 16000 and 8000.
Expected: six reports in `reports/`.

- [ ] **Step 5: Commit**

```bash
git add src/support_agent/voice/bench tests/unit/test_wer.py .gitignore reports
git commit -m "feat: add a speech-to-text benchmark with word error rates"
```

### Task 9: Full pipeline benchmark

**Files:**
- Create: `src/support_agent/voice/bench/pipeline_bench.py`
- Create: `src/support_agent/voice/bench/calls.yaml`
- Modify: `src/support_agent/voice/bench/__main__.py`
- Test: `tests/unit/test_pipeline_bench.py`

**Interfaces:**
- Consumes: `build_task()` (Task 5), `TurnTimer` (Task 4), the audio maker (Task 8).
- Produces: `FileAudioTransport(turns: list[bytes], sample_rate: int)`, a Pipecat transport for tests. It plays each caller turn as audio, then silence, and it waits for the agent audio to end before it plays the next turn. It records the agent audio.
- Produces: `calls.yaml` with three scripted calls: a product question, a guest order status with order number and phone, and a guest order with a yes. Each turn has the caller text and the expected tool names, the same checks as the chat benchmark cases.
- Produces: `uv run python -m support_agent.voice.bench calls [--rate 16000|8000]` writes `reports/<date>-voice-calls-<rate>.md`. The report has these parts:
  1. The pause for each turn, and the median and p90 pause for turns with tools and without tools.
  2. The filler start time.
  3. The pass or fail of each call.
  4. The peak memory of the Ollama process and the agent process.

- [ ] **Step 1: Write the failing test**

```python
async def test_file_transport_plays_turns_in_order():
    # A pipeline of FileAudioTransport and a fake processor that answers each audio turn with
    # one OutputAudioRawFrame -> the processor sees turn 1 audio before turn 2 audio,
    # and the transport records two answers.
```

- [ ] **Step 2: Run the test and make sure that it fails**

Run: `uv run pytest tests/unit/test_pipeline_bench.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement the transport and the command**

The caller audio comes from Kokoro with a voice that is not the agent voice.

- [ ] **Step 4: Run the tests and the benchmark**

Run: `uv run pytest -m "not store and not llm"` and then `uv run python -m support_agent.voice.bench calls --rate 16000` and `--rate 8000`
Expected: all tests pass, and two reports in `reports/`.

- [ ] **Step 5: Commit**

```bash
git add src/support_agent/voice/bench tests/unit/test_pipeline_bench.py reports
git commit -m "feat: add a full voice pipeline benchmark"
```

### Task 10: Documentation and the starting numbers

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `.env.example`
- Create: `reports/2026-11-15-voice-baseline.md`

- [ ] **Step 1: Write the baseline report**

Use the reports of Tasks 8 and 9. Show the pause median and p90. Show the share of each step in the pause: speech end, Whisper, first model token, and first audio. Show the Whisper error rates at 16 kHz and 8 kHz, and the memory. Name the slowest step. Pick the Whisper model with the lowest time that stays at 10 percent error or less, and set it as the `voice_stt_model` default.

- [ ] **Step 2: Update the README and CLAUDE.md**

Add these parts:

- the voice page and the Linphone setup steps
- the Asterisk start command and the voice benchmark commands
- the new configuration values and the voice parts of the architecture

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md .env.example reports src/support_agent/config.py
git commit -m "docs: add the voice setup and the baseline voice measurements"
```

### Task 11: Speed fix: no thinking after tool results

The first candidate from the model comparison report. If the baseline report names another step as the slowest, fix that step first, with the same test and benchmark steps.

**Files:**
- Modify: `src/support_agent/loop.py`, `src/support_agent/config.py`
- Test: `tests/unit/test_loop.py`

**Interfaces:**
- Consumes: `LlmClient.stream(..., reasoning_effort=)` from Task 1.
- Produces: setting `llm_reasoning_effort_after_tools: str = ""`. If it is not empty, `run_turn()` uses it for every model call after a tool result in the same turn. The first call of a turn keeps `llm_reasoning_effort`.

- [ ] **Step 1: Write the failing test**

```python
async def test_effort_after_tools_applies_after_tool_round():
    # settings effort "low", after_tools "none". FakeLlm with one tool round and one answer.
    # -> FakeLlm.efforts_seen == [None, "none"]  (None means the configuration value)
```

- [ ] **Step 2: Run it and make sure that it fails**

Run: `uv run pytest tests/unit/test_loop.py::test_effort_after_tools_applies_after_tool_round -v`
Expected: FAIL.

- [ ] **Step 3: Implement the setting**

`run_turn()` gets a `reasoning_effort_after_tools: str = ""` argument. The server, the voice processor, and the chat benchmark pass the setting.

- [ ] **Step 4: Run the correctness gate**

Run: `uv run pytest -m "not store and not llm"`, then `LLM_REASONING_EFFORT_AFTER_TOOLS=none uv run python -m support_agent.bench --model qwen3.5:4b-q4_K_M`, then `uv run python -m support_agent.voice.bench calls --rate 16000` with the same setting.
Expected: the chat benchmark passes at least 25 of 26, the same as the baseline. If it passes fewer, the setting stays empty, and the report says why.

- [ ] **Step 5: Commit**

```bash
git add src/support_agent tests/unit/test_loop.py reports
git commit -m "perf: allow a lower thinking level after tool results"
```

### Task 12: Comparison report

**Files:**
- Create: `reports/2026-11-15-voice-comparison.md`
- Modify: `README.md`

- [ ] **Step 1: Write the comparison**

Compare the baseline and the result of Task 11: pause median and p90 with and without tools, filler start time, chat benchmark pass rate, Whisper choice, and memory. State which success numbers of the spec pass and which do not, and the next step for each one that does not pass.

- [ ] **Step 2: Add the main numbers to the README and commit**

```bash
git add reports README.md
git commit -m "docs: add the voice speed comparison"
```
