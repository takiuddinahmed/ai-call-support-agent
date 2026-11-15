# Voice comparison, 2026-11-15

This report compares four 16 kHz runs of the full call benchmark. It states which success numbers of the spec pass and which do not.

Each column is one run with 5 calls and 34 turns. The runs vary a lot from one run to the next. Treat a gap of 0.3 s or less as noise. The caller audio is synthetic speech with no noise.

## Result

- The final default is the Whisper turbo model plus the part A fix of Task 11. It is the fourth column.
- Part A fixed the empty answers. The count went from 8 to 0.
- The final run passed 3 of 5 calls. The baseline passed 2 of 5.
- The pause median on turns without tools fell from 3.37 s to 2.99 s. The goal is 1.5 s or less. The goal is still missed.
- The filler start median fell from 2.32 s to 1.98 s. The goal is 1.5 s or less. The goal is still missed.
- The memory upper figure is 9.1 GB. The goal of 12 GB or less is met.

## The four runs

| Column | Run | Report |
|---|---|---|
| Baseline | Whisper turbo, no Task 11 change | `2026-11-15-voice-calls-16000.md` |
| Part A (small) | Prompt budget and fallback line, Whisper small | `2026-11-15-voice-calls-16000-task11-a.md` |
| Part A plus "none" (small) | Part A, and `LLM_REASONING_EFFORT_AFTER_TOOLS=none`, Whisper small | `2026-11-15-voice-calls-16000-task11-b-none.md` |
| Final | Part A, Whisper turbo, after-tools level empty | `2026-11-15-voice-calls-16000-task12-final.md` |

The final run is the only run with the final default pair. It ran on the same Mac, with the same model `qwen3.5:4b-q4_K_M` at `reasoning_effort=low`.

## Comparison

| Measure | Baseline (turbo) | Part A (small) | Part A plus "none" (small) | Final (turbo plus part A) |
|---|---|---|---|---|
| Pause median, turns without tools | 3.37 s | 2.69 s | 2.73 s | 2.99 s |
| Pause p90, turns without tools | 4.26 s | 3.29 s | 4.03 s | 3.60 s |
| Filler start median | 2.32 s | 1.71 s | 1.58 s | 1.98 s |
| Filler start p90 | 7.34 s | 2.27 s | 1.96 s | 3.77 s |
| Answer start median, turns with tools | 6.36 s | 4.23 s | 3.11 s | 4.65 s |
| Empty answers | 8 | 0 | 0 | 0 |
| Calls passed | 2 of 5 | 1 of 5 | 2 of 5 | 3 of 5 |
| Chat benchmark pass rate | 25 of 26 | 25 of 26 | 24 and 25 of 26 (two runs) | 25 of 26 (part A, empty level) |
| Memory upper figure | 13.3 GB | 8.5 GB | 8.5 GB | 9.1 GB |
| Whisper model | `whisper-large-v3-turbo` | `whisper-small-mlx` | `whisper-small-mlx` | `whisper-large-v3-turbo` |

Notes on the table:

- The chat pass rate of the baseline column comes from the model comparison report. The other values come from Task 11. No chat run used exactly the final pair, but Whisper does not take part in the chat benchmark.
- The turn mix differs between runs. The count of turns without tools was 19, 18, 20, and 14. The final run had more turns with tools, so its median covers fewer turns.
- The memory upper figure counts the agent, the runner, and the Ollama estimate. The runner can grow with age. The baseline runner was older, so the baseline figure is high.
- The "none" setting is not in the final default. It made the model skip `prepare_order` in the chat benchmark. See `reports/2026-11-15-task-11-context-and-effort.md`.

## Whisper

The caller said the phone number "0 1 8 1 1 1 1 1 1 1 1". In 3 calls of each small run, the small model heard 10 digits. The turbo model heard 11 digits in the final run, and the order lookups worked. So the default is turbo. The gain of the small model is about 0.12 s.

The turbo model still heard "Sillet" for "Sylhet" in the final run. The order summary then had the wrong city.

## Success numbers of the spec

| Number | Goal | Final result | Verdict |
|---|---|---|---|
| Pause median, turns without tools | 1.5 s or less | 2.99 s | does not pass |
| Filler start, turns with tools | within 1.5 s | median 1.98 s | does not pass |
| Speech-to-text error | 10 percent or less | 0.059 normalized, turbo, 16 kHz | passes |
| Memory | under 12 GB | 9.1 GB upper figure | passes |
| Correctness | same as the chat benchmark | 3 of 5 calls passed | does not pass |

The speech-to-text figure comes from the baseline report of 128 clips. It is a normalized error, not a raw word error rate.

## Failures in the final run

- `voice-guest-order-yes`, turn 4: Whisper heard "Sillet" for "Sylhet". The order summary had the wrong city.
- `voice-natural-long-turns`, turn 2: the caller said the phone number in groups. The turn split in two. The agent answered the first part, and it spoke over the caller. The answer had no status word.
- The run had 2 split turns, and in both the agent spoke over the caller.

The final run did not repeat the price error, but I did not check each price. In run A of Task 11, the agent said 550 taka for the Arduino Uno. The store price is 850. This is a correctness bug of the model, not a speed problem.

## Next step for each number that does not pass

1. Pause median. The first model token takes about 70 percent of the pause. Options: a faster model path or a smaller thinking budget on turns without tools. A lower level after tool results is rejected. It made the model skip `prepare_order`.
2. Filler start. The same first token step sets it. The filler cannot start before the model chooses a tool. One option is to start a filler at the speech end for turns that likely need a tool.
3. Correctness, split turns. Split a turn on grouped phone numbers. Wait longer after a pause that follows a digit group.
4. Correctness, the invented price. Treat the 550 taka answer for the Arduino Uno as a bug. Add a check that each price in an answer comes from a tool result.
5. Correctness, the city. Add a spelling step for city names from a list of the store cities.
6. Memory of long calls. A model context of 8192 tokens keeps more history. It costs about 0.2 GB more memory. Use `MAX_PROMPT_TOKENS=6400` with it.
7. Measurement. Single runs vary a lot. Run the call benchmark 3 times for each change before a decision.
