# Speech-to-text benchmark: mlx-community/whisper-large-v3-turbo at 8000 Hz (G.711 fallback, band-limit only, no codec noise)

Date: 2026-11-15. Clips: 128. Language: English.

## Word error rate

The main number is the normalized word error rate. The normalization makes the
format of numbers and codes the same on both sides (for example ESP32 and ESP 32).
The raw rate counts those format differences as errors.

The mean normalized word error rate is 0.059. The mean raw word error rate is 0.082.

| Sentence group | Normalized | Raw | Clips |
|---|---|---|---|
| product | 0.072 | 0.053 | 36 |
| order | 0.045 | 0.108 | 24 |
| phone | 0.013 | 0.146 | 16 |
| address | 0.154 | 0.154 | 24 |
| yesno | 0.000 | 0.000 | 28 |

| Voice | Normalized | Raw | Clips |
|---|---|---|---|
| af_heart | 0.059 | 0.080 | 32 |
| am_adam | 0.077 | 0.102 | 32 |
| Samantha | 0.043 | 0.074 | 32 |
| Daniel | 0.058 | 0.073 | 32 |

## Time

Each Whisper call ran alone, after the model loaded and one warm-up call.
The times are Whisper time only. They do not include audio capture, voice activity
detection, or speech synthesis. Ollama had a model loaded during the timing, so
the times can have a small error.

- Median time: 0.186 s
- p90 time: 0.194 s

## Worst clips

| Word error rate | Voice | Reference | Heard |
|---|---|---|---|
| 0.60 | Daniel | Send it to Agrabad, Chittagong. | Send it to the grabberd, Chittergong. |
| 0.40 | af_heart | Send it to Agrabad, Chittagong. | Send it to Agrabahd, Chittagang. |
| 0.40 | am_adam | Send it to Agrabad, Chittagong. | Send it to Agrabahd, Chiragung. |
| 0.40 | Samantha | Send it to Agrabad, Chittagong. | Send it to a grabout. Chittagong. |
| 0.33 | am_adam | I need an L298N motor driver. | I need an L298 and motor driver. |
| 0.29 | af_heart | I live in Uttara, Sector 7, Dhaka. | I live in Attara, Sector 7, DACA. |
| 0.25 | am_adam | Deliver to Gulshan 2, Dhaka, near the circle. | Delivered to Gulshan 2, DACA near the circle. |
| 0.25 | af_heart | How much is the HC-SR04 ultrasonic sensor? | How much is the HCSR-04 ultrasonic sensor? |

## Note

Differences of about 0.01 in word error rate are inside the noise of 128
clips. Do not rank models on a difference that small.

The 8 kHz clips are a G.711 fallback test. They are band-limited only and have no
G.711 codec noise. The main phone path is 16 kHz.

Synthetic voices give lower error rates than real callers. Real callers have noise,
accents, and a poor phone line. Use these numbers to compare models, not to predict
the error rate of a real call.
