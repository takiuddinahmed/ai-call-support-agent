# Speech-to-text benchmark: mlx-community/whisper-small-mlx at 8000 Hz (G.711 fallback, band-limit only, no codec noise)

Date: 2026-11-15. Clips: 128. Language: English.

## Word error rate

The main number is the normalized word error rate. The normalization makes the
format of numbers and codes the same on both sides (for example ESP32 and ESP 32).
The raw rate counts those format differences as errors.

The mean normalized word error rate is 0.071. The mean raw word error rate is 0.084.

| Sentence group | Normalized | Raw | Clips |
|---|---|---|---|
| product | 0.061 | 0.041 | 36 |
| order | 0.065 | 0.121 | 24 |
| phone | 0.025 | 0.110 | 16 |
| address | 0.205 | 0.192 | 24 |
| yesno | 0.000 | 0.000 | 28 |

| Voice | Normalized | Raw | Clips |
|---|---|---|---|
| af_heart | 0.064 | 0.075 | 32 |
| am_adam | 0.071 | 0.082 | 32 |
| Samantha | 0.071 | 0.089 | 32 |
| Daniel | 0.078 | 0.089 | 32 |

## Time

Each Whisper call ran alone, after the model loaded and one warm-up call.
The times are Whisper time only. They do not include audio capture, voice activity
detection, or speech synthesis. Ollama had a model loaded during the timing, so
the times can have a small error.

- Median time: 0.068 s
- p90 time: 0.082 s

## Worst clips

| Word error rate | Voice | Reference | Heard |
|---|---|---|---|
| 0.60 | Samantha | Send it to Agrabad, Chittagong. | Send it to a grab at Chittagong. |
| 0.40 | af_heart | Send it to Agrabad, Chittagong. | Send it to Agrabod, Chidegang. |
| 0.40 | Daniel | Send it to Agrabad, Chittagong. | Send it to a grabbered, Chittagong. |
| 0.38 | Samantha | My order number is 3015 and my email is rahim at example dot com. | My order number is 3015 and my email is raheematexample.com |
| 0.38 | Daniel | My order number is 3015 and my email is rahim at example dot com. | My order number is 3015 and my email is raheematexample.com |
| 0.33 | Daniel | Please check order 5120 for me. | Please check order 5124 me. |
| 0.33 | Samantha | My address is House 8, Road 3, Banani, Dhaka. | My address is House 8, Rd 3, Bannany, Dhaka. |
| 0.33 | Daniel | My address is House 8, Road 3, Banani, Dhaka. | My address is House 8, Rd 3, Bannany, Dhaka. |

## Note

Differences of about 0.01 in word error rate are inside the noise of 128
clips. Do not rank models on a difference that small.

The 8 kHz clips are a G.711 fallback test. They are band-limited only and have no
G.711 codec noise. The main phone path is 16 kHz.

Synthetic voices give lower error rates than real callers. Real callers have noise,
accents, and a poor phone line. Use these numbers to compare models, not to predict
the error rate of a real call.
