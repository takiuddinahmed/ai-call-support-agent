# Speech-to-text benchmark: mlx-community/distil-whisper-large-v3 at 16000 Hz

Date: 2026-11-15. Clips: 128. Language: English.

## Word error rate

The main number is the normalized word error rate. The normalization makes the
format of numbers and codes the same on both sides (for example ESP32 and ESP 32).
The raw rate counts those format differences as errors.

The mean normalized word error rate is 0.096. The mean raw word error rate is 0.112.

| Sentence group | Normalized | Raw | Clips |
|---|---|---|---|
| product | 0.108 | 0.119 | 36 |
| order | 0.052 | 0.091 | 24 |
| phone | 0.147 | 0.203 | 16 |
| address | 0.199 | 0.190 | 24 |
| yesno | 0.000 | 0.000 | 28 |

| Voice | Normalized | Raw | Clips |
|---|---|---|---|
| af_heart | 0.088 | 0.105 | 32 |
| am_adam | 0.114 | 0.120 | 32 |
| Samantha | 0.089 | 0.113 | 32 |
| Daniel | 0.093 | 0.109 | 32 |

## Time

Each Whisper call ran alone, after the model loaded and one warm-up call.
The times are Whisper time only. They do not include audio capture, voice activity
detection, or speech synthesis. Ollama had a model loaded during the timing, so
the times can have a small error.

- Median time: 0.176 s
- p90 time: 0.185 s

## Worst clips

| Word error rate | Voice | Reference | Heard |
|---|---|---|---|
| 0.60 | Daniel | Send it to Agrabad, Chittagong. | Send it to a grab-od. Chittagong. |
| 0.40 | af_heart | Send it to Agrabad, Chittagong. | Send it to Agribod, Chidegang. |
| 0.40 | am_adam | Send it to Agrabad, Chittagong. | Send it to Agrobod Chidagung. |
| 0.40 | Samantha | Send it to Agrabad, Chittagong. | Send it to a grabout, chittagong. |
| 0.38 | am_adam | Deliver to Gulshan 2, Dhaka, near the circle. | delivered to Gulsion 2, DACA near the circle. |
| 0.33 | am_adam | I need an L298N motor driver. | I need an L-298 and motor driver. |
| 0.31 | Daniel | Is the ESP32 DevKit better than the Arduino Uno for Wi-Fi projects? | Is the Sp 32 dev kit better than the Arduino for Wi-Fi projects? |
| 0.30 | Samantha | I want two L298N motor drivers and one Raspberry Pi 5. | I want 2 L-298N motor drivers and 1 Raspberry Pi 5. |

## Note

Differences of about 0.01 in word error rate are inside the noise of 128
clips. Do not rank models on a difference that small.

The 8 kHz clips are a G.711 fallback test. They are band-limited only and have no
G.711 codec noise. The main phone path is 16 kHz.

Synthetic voices give lower error rates than real callers. Real callers have noise,
accents, and a poor phone line. Use these numbers to compare models, not to predict
the error rate of a real call.
