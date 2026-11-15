# Voice baseline, 2026-11-15

This report gives the starting numbers of the voice agent. All numbers come from the reports of the speech-to-text benchmark and the full call benchmark. Both ran on the Apple M5 Pro with 24 GB.

The call numbers come from one run for each sample rate. Each run has 5 calls and 34 turns. Run the benchmark more than one time before you compare two changes. The caller audio is synthetic speech with no noise. A real caller gives more errors.

## Result

- The pause median is 3.37 s on turns without tools. The goal is 1.5 s or less. The goal is missed.
- The model is the slowest step. The first model token takes about 70 percent of the pause.
- Only 2 of 5 calls passed. The calls found real problems. See "Findings".
- The Whisper default is now `mlx-community/whisper-small-mlx`.

## Pause

The pause is the time from the end of the caller audio to the first agent audio. On a turn with tools, the first agent audio is the filler phrase. The answer start is the time to the answer audio.

| Rate | Turns | Count | Pause median (s) | Pause p90 (s) | Answer median (s) | Answer p90 (s) |
|---|---|---|---|---|---|---|
| 16 kHz | Without tools | 19 | 3.37 | 4.26 | 3.37 | 4.26 |
| 16 kHz | With tools (filler) | 15 | 2.32 | 7.34 | 6.36 | 12.85 |
| 16 kHz | All turns | 34 | 3.15 | 5.96 | 4.11 | 9.34 |
| 8 kHz | Without tools | 20 | 2.83 | 3.39 | 2.83 | 3.39 |
| 8 kHz | With tools (filler) | 14 | 2.86 | 3.86 | 5.86 | 9.36 |
| 8 kHz | All turns | 34 | 2.83 | 3.83 | 3.60 | 7.52 |

The 16 kHz rate is the main path. It is used by the browser and by the G.722 phone. The 8 kHz run is a G.711 test with a band limit only and no codec noise.

Goals of the spec:

| Goal | Target | 16 kHz | 8 kHz |
|---|---|---|---|
| Pause median, turns without tools | 1.5 s or less | 3.37 s, missed | 2.83 s, missed |
| Filler start median, turns with tools | 1.5 s or less | 2.32 s, missed | 2.86 s, missed |
| Memory upper figure | 12 GB or less | 13.3 GB, missed | 13.7 GB, missed |
| Calls passed | all | 2 of 5, missed | 2 of 5, missed |

## Share of each step in the pause

The call benchmark records the pause of each turn, but not the time of each step. The step times below come from one real browser call without tools. The debug panel measured them. Its pause was 3148 ms. This agrees with the 3.37 s median of the benchmark.

| Step | Time (ms) | Share of the pause |
|---|---|---|
| Speech end (voice activity and turn-end decision) | 59 | 2 percent |
| Whisper (speech-to-text) | 149 | 5 percent |
| First model token | 2214 | 70 percent |
| First model token to first audio (the first sentence and Kokoro) | 726 | 23 percent |

The first row is the time before the turn-end decision. The other rows are the rest of the pause. The shares come from one turn, so treat them as a rough picture. A turn with tools is slower, because the model first writes the tool call. In a browser turn with a tool, the first token took 4.3 s.

The slowest step is the first token of the Qwen model.

The two Whisper steps (speech end and Whisper) together take about 7 percent of the pause. A faster Whisper model saves about 0.12 s. It cannot reach the 1.5 s goal.

## Whisper

All Whisper numbers use 128 synthetic clips. The voices are Kokoro and macOS `say`. The normalization makes the format of numbers and codes the same on both sides. The time is the Whisper time only.

| Model | Normalized error 16 kHz | Normalized error 8 kHz | Median time 16 kHz (s) | p90 time 16 kHz (s) |
|---|---|---|---|---|
| `mlx-community/whisper-large-v3-turbo` | 0.059 | 0.059 | 0.186 | 0.196 |
| `mlx-community/whisper-small-mlx` | 0.064 | 0.071 | 0.069 | 0.081 |
| `mlx-community/distil-whisper-large-v3` | 0.096 | 0.102 | 0.176 | 0.185 |

The rule is to pick the fastest model with a normalized error of 0.10 or less. The small model passes at both rates and is the fastest. So `voice_stt_model` now defaults to `mlx-community/whisper-small-mlx`.

The gap between the small model (0.064) and the turbo model (0.059) is inside the noise of 128 clips. Phone numbers (0.147 for distil) and addresses (0.199 for distil) are the hardest groups.

The full call benchmark ran with the turbo model, because the default changed after the run. Run it again with the small model to confirm the numbers.

## Memory

| Rate | Memory upper figure |
|---|---|
| 16 kHz | 13.3 GB |
| 8 kHz | 13.7 GB |

The upper figure is the agent footprint, plus the runner footprint, plus the Ollama estimate. The last two can count the same weights two times. So the real memory is lower than the upper figure. At 16 kHz the peaks were 4.3 GB for the agent and 5.85 GB for the runner. The goal is 12 GB or less.

## Findings

The call benchmark found these problems:

1. Context overflow. The Ollama context is 4096 tokens. A long call fills it. Then the model gives an empty answer and the caller hears nothing. The 20-turn call gave empty answers in the late turns. The 16 kHz run had 8 empty answers.
2. Split turns. The turn-end model ends a turn at a pause inside a sentence. This happens, for example, when a caller says a phone number in groups. The agent answers the first part and speaks over the caller. This run had 2 split turns and 2 turns where the agent spoke over the caller.
3. Wrong city. Whisper heard "Sillet" for "Sylhet". The order summary then had the wrong city.
4. The guest order call often fails. The call `voice-guest-order-yes` failed in the runs. The agent repeated tool calls at the last turn.
5. The memory upper figure of 13.3 GB can count the same weights two times. Do not read it as the real memory.

## Next steps

1. Task 11 first fixes the context overflow. Correctness wins over speed.
2. Task 11 then tries a lower thinking level after tool results. The model produces the first token slowly, so this is the best chance to reduce the pause.
3. Split-turn tuning comes after these two steps.
4. Run the full call benchmark again with the small Whisper model, and more than one time.
