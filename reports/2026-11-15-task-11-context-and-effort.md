# Task 11: context overflow fix and thinking level after tools, 2026-11-15

This report gives the cause of the empty answers in long calls, the fix, and the test of a lower thinking level after tool results. All runs used `qwen3.5:4b-q4_K_M` in Ollama 0.40.16 on the Apple M5 Pro with 24 GB.

## Result

- Empty answers: 8 before, 0 after, in the 16 kHz call benchmark. In a text replay of the 20-turn call, 11 of 20 turns were empty before, and 0 after.
- The agent keeps each prompt under `MAX_PROMPT_TOKENS=3200` estimated tokens. The Ollama context stays at 4096 tokens.
- `LLM_REASONING_EFFORT_AFTER_TOOLS=none` is rejected. In two chat benchmark runs, the model skipped `prepare_order` and wrote its own order summary. The default stays empty.

## Cause of the empty answers

The Ollama runner log (`~/.ollama/logs/server.log`) shows the runner command `llama-server ... -c 4096 --context-shift --keep 4`. It also shows "KV cache shifting is not supported for this context, disabling KV cache shifting". So the runner cannot move old tokens out during generation.

A text replay of the 20-turn call `voice-long-call-memory` recorded `usage.prompt_tokens` of each model call. It ran with the old code.

| Turn | Prompt tokens | Completion tokens | Visible answer |
|---|---|---|---|
| 1 | 1849 | 94 | yes |
| 8 | 3935 | 140 | yes |
| 9 | 4022 | 74 | empty |
| 12 | 4084 | 12 | empty |
| 15 | 4094 | 2 | empty |
| 16 | 3658 | 112 | yes |
| 17 | 4019 | 77 | empty |
| 20 | 4064 | 32 | empty |

From turn 9, the prompt tokens plus the completion tokens are 4096. The model starts with hidden thinking. The runner stops at the end of the context, before the first visible token. The runner log shows the same stop: "stop processing: n_tokens = 4095, truncated = 1".

At turn 16, the prompt tokens go down from 4094 to 3658, but the prompt characters go up. Ollama cut the prompt. A probe with a system code word and 40 filler turns (about 33,000 characters) found these facts:

- Ollama keeps the system message and drops the oldest messages. The model still knew the code word, but it forgot the name from the first turn.
- Ollama fills the context to about 4020 tokens. So almost no room stays for the thinking and the answer.

So the system prompt was not lost. The cause is that the prompt fills the context, and the hidden thinking then uses the last free tokens. The Ollama cut also drops messages without a rule. It can split a tool call from its tool result, and it can drop the pending order summary.

## Fix

`src/support_agent/history.py` keeps the prompt under `max_prompt_tokens`. The estimate is ASCII characters / 3 plus other characters / 1.5. The measured ratio was 3.5 to 4.3 characters for each token in the agent JSON, and 1.9 for Bangla text. So the estimate is a little too high. If the prompt is too large, the agent does these steps:

1. It replaces the tool results of older turns with a short note. The note tells the model to call the tool again.
2. It drops the oldest whole customer turns, but never the current turn.

A tool call and its result stay together. The tool result with the pending order summary ID stays. The trim goes down to the middle of the free room. Then the start of the prompt stays the same for some turns, and the runner can use its prompt cache. `session.messages` does not change.

If a model call gives no text and no tool call, the agent says "Sorry, I did not get that. Can you say it again?". The event stream has an `error` event with `fallback: true`.

After the fix, the same text replay had no empty answers. The largest prompt was 2574 tokens. At turn 17, the agent used the order number from turn 16.

## Context size choice

The 4096 context stays. A variant with `PARAMETER num_ctx 8192` was measured:

| Context | `ollama ps` size | Runner footprint in a 20-turn replay | Empty answers in the replay, without the trim |
|---|---|---|---|
| 4096 | 3.2 GB | not measured in that replay | 11 of 20 |
| 8192 | 3.4 GB | 2.16 GB peak | 0 of 20 (largest prompt 5398 tokens) |

The larger context costs about 0.2 GB. The reasons to keep 4096 follow:

- The trim fixes the empty answers for every context size. A 900-second call can fill 8192 tokens too.
- Shorter prompts give a faster first token after a prompt cache miss.
- A larger context needs a model variant or an Ollama server setting. That is one more setup step.

README "Model context" tells how to use the 8192 variant with `MAX_PROMPT_TOKENS=6400`.

## Thinking level after tool results

The setting `LLM_REASONING_EFFORT_AFTER_TOOLS` sets the thinking level of each model call after a tool result in the same turn. The first model call keeps `LLM_REASONING_EFFORT`.

| Run | After-tools level | Passed | First token median | Turn median | Failures |
|---|---|---|---|---|---|
| `2026-11-15-1949` | empty (low) | 25 of 26 | 1.81 s | 5.44 s | place-en-guest-change-city |
| `2026-11-15-1951` | none | 24 of 26 | 1.81 s | 3.29 s | orders-bn-status-logged-in, place-bn-guest-yes |
| `2026-11-15-1953` | none | 25 of 26 | 1.69 s | 2.96 s | place-bn-guest-yes |

With `none`, the model skipped `prepare_order` in `place-bn-guest-yes` in both runs. It wrote its own order summary after `search_products`. The instructions forbid a model-made summary. So the default stays empty, and `none` is rejected.

## Call benchmark at 16 kHz

The "before" run is `2026-11-15-voice-calls-16000.md` (Whisper turbo). The "after" runs used Whisper small, the new default. So the Whisper change and the trim both change the pause.

| Measure | Before | After (part A) | After, effort none after tools |
|---|---|---|---|
| Report | `voice-calls-16000.md` | `voice-calls-16000-task11-a.md` | `voice-calls-16000-task11-b-none.md` |
| Empty answers | 8 | 0 | 0 |
| Calls passed | 2 of 5 | 1 of 5 | 2 of 5 |
| Pause median, turns without tools | 3.37 s | 2.69 s | 2.73 s |
| Filler start median | 2.32 s | 1.71 s | 1.58 s |
| Answer start median, turns with tools | 6.36 s | 4.23 s | 3.11 s |
| Memory upper figure | 13.3 GB | 8.5 GB | 8.5 GB |

The part A failures are not from the trim:

- Whisper small heard "0181111111" (10 digits) for "0 1 8 1 1 1 1 1 1 1 1" in 3 calls. Then the order lookup failed, or the agent asked for the phone again. The turbo run heard 11 digits.
- In `voice-long-call-memory` turn 2, the model said five hundred fifty taka for the Arduino Uno. The store price is 850. The prompt was far under the budget at that turn, so the trim did not remove a fact.

The memory upper figure is lower, because the runner footprint was 2.0 GB and not 5.85 GB. The runner prompt cache grows during a long Ollama session (limit 8 GiB in the runner log). So the memory figure depends on the age of the runner.

With `none` after tools, the answer start on turns with tools went from 4.23 s to 3.11 s. The pause on turns without tools did not change, because those turns have no model call after a tool result. The call `voice-guest-order-yes` failed again, because Whisper small heard the phone number with 10 digits. The voice calls give no proof against `none`, but the chat benchmark does. So `none` stays rejected.

The memory figures and the pass counts come from one run each. Run the benchmark more than one time before you compare two changes.
