# Model comparison, 2026-11-15

This report compares five local models on the 26 benchmark conversations. All runs used the same agent code (commit `50fd583`), the Apple M5 Pro with 24 GB, and Ollama 0.40.16.

## Result

The default model is now `qwen3.5:4b-q4_K_M` with `LLM_REASONING_EFFORT=low`.

It passed 25 of 26 conversations (96 percent), the same as the best model. Its speed is the same as the other 4B builds, and its memory stays at about 4 GB. The MLX build of the same model grew to 15 GB during a run.

## Correctness and speed

| Model | Runner | Pass | Bangla | English | Mixed | First token median | First token p90 | Turn median | Turn p90 | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.5:4b` | MLX | 25/26 (96%) | 4/5 | 17/17 | 4/4 | 1.9 s | 7.5 s | 7.2 s | 20.0 s | 66 |
| `qwen3.5:4b-q4_K_M` | llama.cpp | 25/26 (96%) | 5/5 | 16/17 | 4/4 | 2.1 s | 5.1 s | 6.9 s | 11.9 s | 57 |
| `qwen3.5:4b-mtp-q4_K_M` | llama.cpp | 24/26 (92%) | 4/5 | 16/17 | 4/4 | 2.0 s | 4.3 s | 6.3 s | 14.9 s | 60 |
| `qwen3.5:9b-q4_K_M` | llama.cpp | 24/26 (92%) | 4/5 | 16/17 | 4/4 | 3.3 s | 6.5 s | 8.2 s | 16.4 s | 33 |
| `gemma4:e4b-it-q4_K_M` | llama.cpp | 24/26 (92%) | 5/5 | 16/17 | 3/4 | 2.2 s | 4.5 s | 5.2 s | 11.1 s | 87 |

The Qwen models ran with `reasoning_effort=low`. Gemma ran with the model default, because it has no thinking levels.

"First token" is the time from the customer message to the first visible word of a model call. "p90" is the slowest 10 percent. "Turn" is the full customer turn, with all tool calls.

## Memory

The automatic memory columns of the benchmark are not reliable. The Ollama estimate adds up every loaded model, and the chat app had a second model loaded during some runs. The macOS process footprint does not count model weights that the llama.cpp runner maps from disk. So the memory was measured again, one model at a time.

| Model | System memory after load | Ollama estimate after load | Peak during a full benchmark run |
| --- | --- | --- | --- |
| `qwen3.5:4b` (MLX) | 4.4 GB | 4.1 GB | 15.2 GB (process footprint) |
| `qwen3.5:4b-q4_K_M` | 4.2 GB | 3.2 GB | no growth seen |
| `qwen3.5:4b-mtp-q4_K_M` | 2.8 GB | 1.3 GB | no growth seen |
| `qwen3.5:9b-q4_K_M` | 4.9 GB | 4.8 GB | no growth seen |
| `gemma4:e4b-it-q4_K_M` | 3.6 GB | 0.3 GB | no growth seen |

"System memory after load" is the drop in free macOS memory after the first answer, with a context of 4096 tokens. The MLX runner keeps a growing cache, and its footprint passed the 12 GB budget during each long run.

## Failures

| Model | Failed conversations |
| --- | --- |
| `qwen3.5:4b` | `place-bn-guest-yes`: showed the summary, but did not place the order after "হ্যাঁ, অর্ডার করে দিন". |
| `qwen3.5:4b-q4_K_M` | `refuse-en-cancel`: did not say that it cannot cancel an order. |
| `qwen3.5:4b-mtp-q4_K_M` | `place-bn-guest-yes` and `refuse-en-cancel`, as above. |
| `qwen3.5:9b-q4_K_M` | `place-en-guest-yes`: summary without the total 270. `place-bn-guest-yes`: no `prepare_order` call. |
| `gemma4:e4b-it-q4_K_M` | `stock-en-stm32` and `stock-mixed-orange-pi`: did not say that the product is out of stock. |

No model placed an order without a "yes", showed an order without proof, or invented a product.

## Choice

The plan rule picks the fastest model that passes at least 90 percent and stays under 12 GB.

- The MLX build `qwen3.5:4b` fails the memory rule.
- The other four models pass the rule.
- The first token medians of the three 4B Qwen builds and Gemma are within 0.3 s of each other. The difference is smaller than the noise of these runs, because the chat app also used the model server during the runs.
- `qwen3.5:4b-q4_K_M` has the highest pass rate of the four, and all 5 Bangla conversations pass.

So the default is `qwen3.5:4b-q4_K_M`.

## Gaps to the targets

- Speed: the design target is a first token under 1 second for most turns. All models need about 2 seconds, because `reasoning_effort=low` adds hidden thinking. With `none`, the first token comes in about 0.2 seconds, but the model skips the store tools and invents prices.
- The next optimization steps to test:
  1. Answer simple turns (greetings, a yes or no) with thinking off, and product and order turns with thinking low.
  2. Force a tool call in the first model round of a product question, so that thinking can be off.
  3. Try Gemma with stronger stock rules in the instructions, because it is the fastest per turn.
  4. Set a token limit for each model call. One run of the first attempt hung for over 120 seconds on one turn.

## Raw reports

The per-model reports are in this folder, with the date and the model in the file name.
