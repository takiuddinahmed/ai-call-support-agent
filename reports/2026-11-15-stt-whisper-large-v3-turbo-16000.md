# Speech-to-text benchmark: mlx-community/whisper-large-v3-turbo at 16000 Hz

Date: 2026-11-15. Clips: 128. Language: English.

## Word error rate

The main number is the normalized word error rate. The normalization makes the
format of numbers and codes the same on both sides (for example ESP32 and ESP 32).
The raw rate counts those format differences as errors.

The mean normalized word error rate is 0.059. The mean raw word error rate is 0.086.

| Sentence group | Normalized | Raw | Clips |
|---|---|---|---|
| product | 0.075 | 0.053 | 36 |
| order | 0.038 | 0.107 | 24 |
| phone | 0.000 | 0.158 | 16 |
| address | 0.165 | 0.165 | 24 |
| yesno | 0.000 | 0.000 | 28 |

| Voice | Normalized | Raw | Clips |
|---|---|---|---|
| af_heart | 0.070 | 0.095 | 32 |
| am_adam | 0.072 | 0.097 | 32 |
| Samantha | 0.044 | 0.083 | 32 |
| Daniel | 0.051 | 0.067 | 32 |

## Time

Each Whisper call ran alone, after the model loaded and one warm-up call.
The times are Whisper time only. They do not include audio capture, voice activity
detection, or speech synthesis. Ollama had a model loaded during the timing, so
the times can have a small error.

- Median time: 0.186 s
- p90 time: 0.196 s

## Worst clips

| Word error rate | Voice | Reference | Heard |
|---|---|---|---|
| 0.40 | af_heart | Send it to Agrabad, Chittagong. | Send it to Agrabod, Chiragang. |
| 0.40 | am_adam | Send it to Agrabad, Chittagong. | Send it to Agrabahd, Chiragung. |
| 0.40 | Samantha | Send it to Agrabad, Chittagong. | Send it to a grabout. Chittagong. |
| 0.40 | Daniel | Send it to Agrabad, Chittagong. | Send it to a grabberd, Chittagong. |
| 0.33 | am_adam | I need an L298N motor driver. | I need an L298 and motor driver. |
| 0.33 | af_heart | My address is Flat 4B, 27 Mirpur Road, Dhaka. | My address is flat4b, 27 Merper Road, Dhaka. |
| 0.30 | Samantha | I want two L298N motor drivers and one Raspberry Pi 5. | I want 2 L298N motor drivers and 1 Raspberry Pi 5. |
| 0.29 | af_heart | I live in Uttara, Sector 7, Dhaka. | I live in Attara, Sector 7, DACA. |

## Note

Differences of about 0.01 in word error rate are inside the noise of 128
clips. Do not rank models on a difference that small.

The 8 kHz clips are a G.711 fallback test. They are band-limited only and have no
G.711 codec noise. The main phone path is 16 kHz.

Synthetic voices give lower error rates than real callers. Real callers have noise,
accents, and a poor phone line. Use these numbers to compare models, not to predict
the error rate of a real call.
