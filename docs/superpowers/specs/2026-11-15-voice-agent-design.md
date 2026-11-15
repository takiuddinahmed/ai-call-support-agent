# Voice support agent design

Date: 2026-11-15

## Intent

The main project is an AI customer service agent that answers voice calls for the demo store Robo Parts BD. The first phase built a text chat agent. Its design is in `2026-11-15-chat-agent-design.md`.

This design covers the second phase: voice calls in English only, on two channels. On the first channel, a person opens a page, starts a call, and talks through the microphone. On the second channel, a person calls from a softphone app on an iPhone. The agent answers with a spoken voice. The agent uses the same core as the chat agent: the instructions, the store tools, the order rules, and the store client.

The user has two reasons for the project:

1. To build a working support agent for calls.
2. To learn how to run AI models locally on limited resources, and how to measure and improve their speed and memory use.

The user chose speed over voice quality for this phase. A slow voice agent feels broken, because callers talk over a long silence or hang up. A plain voice still works, and a better voice model can replace it later.

## Goals

- A voice page at `/voice`. On it, a person logs in or calls as a guest, starts a call, and talks with the agent in English.
- The same three customer needs as the chat agent: find a product, place a cash-on-delivery order, and get the status of an order.
- Natural turn-taking. The agent hears the end of each caller sentence. If the caller talks while the agent speaks, the agent stops and listens.
- No long silence. When the agent starts a store tool call, it says a short phrase at once, such as "Let me check that."
- IP phone calls. A person calls the agent from the Linphone app on an iPhone on the same Wi-Fi network. The call goes through an Asterisk server in Docker. SIP is the standard protocol for internet phone calls.
- A debug panel that shows the time of each step of each turn, as defined in "Timing" below. Phone calls show in the same panel.
- A voice benchmark that measures the speed of the full pipeline and the error rate of speech-to-text on recorded audio.

## Non-goals

- Bangla and mixed-language speech. The chat page keeps Bangla. Bangla voice comes in a later phase.
- A real phone number from a SIP provider. A later phase can connect one to the same Asterisk server.
- Calls from outside the home network.
- A natural-sounding voice. The voice must be clear, but it can sound plain.
- More than one call at a time.
- Any change to the demo store.
- Any change to the order rules or the privacy rules of the chat agent.

## Anti-goals

These results are failures, and a working call does not change that:

- The voice path weakens a rule that the chat agent enforces. The agent places an order without a clear spoken yes, shows an order without proof, or states a fact that no tool returned.
- An interruption places an order twice, or leaves the conversation history in a state that breaks the next model call.
- The time of a step is not visible. The user cannot optimize what the panel does not show.
- The voice code copies the core instead of using it, so a fix in one place does not reach the other.
- The pipeline uses a cloud AI service.

## Constraints

- Hardware: Apple M5 Pro, 24 GB shared memory. The demo store containers use about 3 GB.
- Memory budget: all AI models together use at most 12 GB. The language model `qwen3.5:4b-q4_K_M` uses about 4.2 GB.
- Docker on macOS cannot use the Apple GPU. All models run directly on macOS.
- The language model needs about 2 seconds to its first word with `reasoning_effort=low`. A turn with store tools needs about 7 seconds at the median. These numbers are in `reports/2026-11-15-model-comparison.md`.
- Pipecat 1.16.0 is the current release. It has optional parts for MLX Whisper, Kokoro, Piper, Silero, the local turn-end model, and WebRTC (the real-time audio standard in browsers).

## Approaches

The user picked approach 1.

1. Pipecat in the browser (picked). Pipecat handles the browser audio, the pause detection, and the interruptions. Our agent turn runs as one step of the Pipecat pipeline, so the core does not change. Pipecat also supports phone audio, so the later Asterisk phase reuses most of this work.
2. Our own audio pipeline without Pipecat. The user learns more about each part, but the project must write the pause detection, the interruption logic, and the phone audio code itself. This takes several times more work before the first call works.
3. One speech-to-speech model that hears and talks directly. No local model of this type has the tool calls and the order rules that the agent needs. The later Bangla phase also rules it out, because no local model of this type supports Bangla well.

## Decisions from the user

- English only in this phase.
- Speed first. The voice can be plain.
- Two call channels in this phase: a browser call and an IP phone call through Asterisk.
- The phone is the user's iPhone with a softphone app. Claude proposed Linphone.
- One call at a time.

## Decisions from Claude (the user can change these)

### Project layout

- The voice code goes into the same repository, in `src/support_agent/voice/`.
- The chat server serves the voice page at `/voice` and the WebRTC connection under `/api/voice/`. One process serves chat and voice.
- The voice code imports the core (`loop.py`, `tools.py`, `session.py`, `instructions.py`, `store/`). It does not copy any part of it.
- The builder can make small changes to the core for voice: new events, new hooks, and voice-only rules. The chat behavior and the order and privacy rules of the chat agent must not get weaker. The chat tests must pass.
- The voice models load at the first call and stay loaded. So the chat server starts as fast as before.

### Pipeline

The pipeline has these steps, in this order:

1. Browser transport: Pipecat `SmallWebRTCTransport` receives the microphone audio and sends the agent audio.
2. Speech detection: Silero finds speech in the audio. The local turn-end model ("smart turn") decides the end of each caller sentence.
3. Speech-to-text: MLX Whisper on the Apple GPU, with the language fixed to English.
4. Agent step: a custom Pipecat processor runs `run_turn()` on the session of the call. It sends the answer text to the next step as it streams.
5. Text-to-speech: Kokoro, which is a small English voice model, speaks the text one sentence at a time.
6. Browser transport: the agent audio goes back to the browser.

The first speech models are `whisper-large-v3-turbo` and Kokoro. The speech benchmark decides the final choice between them and the smaller options, as described in "Voice benchmark".

### Agent step

- The session uses `channel="voice"`. The voice instructions must say "Answer in English", in place of the rule "Answer in the language of the customer's last message".
- The agent step speaks a short phrase at once at the start of a tool call. If the agent already spoke answer text in this turn, it skips the phrase. To start the phrase as early as possible, `loop.py` gets a new event on the first streamed piece of a tool call. The `tool_start` event comes only after the full model stream, which is too late. The phrase comes from a fixed list in the code, not from the model. Examples: "Let me check that." and "One moment, please."
- The "Answer in English" rule applies to the voice channel only. The voice style switch in the chat debug panel then also gives English answers. This is accepted.
- The voice instructions keep the voice style: one to three short sentences, no lists, no symbols, and numbers in words.
- The agent says the order number digit by digit, for example "order number one one".

### Interruptions

Caller speech before the first agent audio of a turn does not stop the turn. For example, a caller says "hello?" during a store search. The agent step holds that speech and handles it as the next turn, after the current answer.

When the caller talks while the agent speaks, Pipecat stops the agent audio. The agent step then follows these rules:

- If the model is still writing, the agent step stops the model call.
- If a tool call runs, the agent step lets it finish and keeps its result. This rule protects `place_order`. The order is either placed and recorded in the history, or it is not placed.
- The agent step keeps the history valid for the next model call. Every assistant message with tool calls must have a tool result for each call. If a turn stops before the model answer, the history keeps the customer message and the finished tool results.
- The history keeps only the part of the answer that the caller heard, not the full text that the model wrote. If Pipecat does not report the heard part, the history keeps the sentences that the text-to-speech step started.

### Order approval by voice

On the voice channel, `place_order` accepts a summary only after the caller heard the full summary audio. After the text-to-speech step finishes the answer that contains the summary, the agent step marks the summary as heard. If the caller interrupted the summary, the agent must call `prepare_order` again and read the new summary before a yes counts. A yes that the caller said before the end of the summary audio does not count. Only a customer turn after the end of the summary audio can approve the summary. This rule follows the chat anti-goal that the agent first reads back the items, the address, the delivery option, and the total. The chat channel keeps its current rule.

### IP phone channel

- Asterisk 22 runs in Docker in the `demo-store` compose project, in a new folder `asterisk/`. It registers one SIP account for the iPhone and one number for the agent.
- When the iPhone calls the agent number, Asterisk sends the call audio over a WebSocket (a two-way web connection) to the agent server. The Asterisk WebSocket channel ("chan_websocket") does this. If that channel does not work in the chosen Asterisk image, AudioSocket (a simple Asterisk TCP audio protocol) is the fallback.
- Pipecat has no ready Asterisk adapter. The voice code adds a small one: a frame serializer for the Pipecat WebSocket transport, or a small transport for AudioSocket.
- After the transport, the phone call uses the same pipeline as the browser call: speech detection, Whisper, the agent step, Kokoro, and the debug panel.
- Codec: G.722, which carries 16 kHz audio. G.711 (8 kHz) is the fallback. Linphone supports both. Kokoro audio is resampled to the codec rate.
- Network: the iPhone and the Mac are on the same Wi-Fi. Asterisk publishes the SIP port and a small fixed range of audio ports through Docker. Asterisk announces the LAN address of the Mac, which comes from the configuration.
- Phone callers are guests. Asterisk passes the caller number to the agent, but a SIP number on a home network proves nothing. A phone caller gets an order status with the order number and the phone number of the order, as in a browser guest call.
- The debug panel lists phone calls next to browser calls, so the user can watch the times of a phone call on the Mac.
- The voice benchmark runs the full pipeline test once with the browser audio rate and once with the phone audio rate. The report shows the effect of phone audio on the Whisper error rate.
- The README gets the steps to set up Linphone on the iPhone: the server address, the account, and the password.

### Identity and privacy

- The voice page has the same login form as the chat page. A caller logs in before the call or calls as a guest. The login, the logout, and `start_over()` work as in chat.
- A guest gets an order status with the order number and the phone number of the order. The voice instructions ask for the phone number, not the email, because spoken email addresses are hard for speech-to-text. If the caller says an email, the agent can use it.
- A guest order needs an email in the store. If a voice guest gives no email, the agent step uses an email made from the phone number, in the form `voice-8801XXXXXXXXX@guest.robopartsbd.local`. The caller can then look up the order with the phone number. This rule needs a small change in `prepare_order` or in the voice instructions. The builder picks the place, and the order rules must not change.

### Timing

The debug panel shows these times for each turn, in milliseconds:

- Speech end: the time from the end of the caller speech to the turn-end decision.
- Speech-to-text: the time of the Whisper call.
- The existing agent events: first token, model call time, tool call times.
- First audio: the time from the turn-end decision to the first agent audio sent to the browser.
- Pause: the time from the end of the caller speech to the first agent audio. This is the main number of the phase.

The debug panel also shows the text that Whisper heard and the text that the agent spoke. The page receives the events over the WebRTC data channel or over a server-sent event stream. The builder picks one.

### Voice benchmark

The benchmark has two parts. Both run from the command line and write a report in `reports/`.

1. Speech-to-text: the benchmark runs each Whisper model on a set of recorded English sentences. It reports the word error rate (the share of wrong words) and the time. The sentences cover product names ("ESP32", "Arduino Uno", "L298N"), order numbers, phone numbers, and a yes or no. The first set comes from Kokoro and macOS `say` voices. The user can add own recordings later. Synthetic voices give lower error rates than real callers, and the report must say so.
2. Full pipeline: the benchmark plays recorded caller turns into the pipeline without a browser and measures the pause of each turn. It uses a few scripted calls: a product question, an order status, and an order with a yes.

### Success numbers

Correctness always wins over speed. The pause numbers below are goals for the work of steps 4 and 5 in "Order of work". They are not a gate for the first working call. A speed fix that makes the model skip a tool or invent a fact is rejected.

- Pause: a median of 1.5 seconds or less on turns without a tool call. On turns with a tool call, the filler phrase starts within 1.5 seconds.
- Speech-to-text: a word error rate of 10 percent or less on the benchmark sentences.
- Memory: all models together stay under the 12 GB budget during a 20-turn call.
- Correctness: the scripted voice calls give the same results as the matching chat benchmark cases.

### Order of work

1. Build the pipeline with the first models for the browser call, and show the times in the debug panel.
2. Add Asterisk and the phone transport, and make a call from Linphone.
3. Run the benchmarks and record the starting numbers.
4. Fix the slowest step. The likely first fix is the model thinking time: thinking off for simple turns, as the model comparison report proposes.
5. Run the benchmarks again and write a comparison report.

## Left to the builder

- The Asterisk image, its configuration files, and the account names and passwords for the demo.
- The exact Pipecat classes and their parameters, and how the custom agent processor maps `AgentEvent`s to Pipecat frames.
- The transport of debug events to the page: WebRTC data channel or server-sent events.
- The text split for text-to-speech, as long as one sentence goes to the voice as soon as it is complete.
- The list of filler phrases.
- The format and the storage of the benchmark audio files.
- Unit tests with fakes for the agent step, the interruption rules, and the filler phrase. The existing `FakeLlm` and `FakeStore` stay the base for these tests.
