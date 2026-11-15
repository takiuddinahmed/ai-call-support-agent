# Speech-to-text benchmark: mlx-community/distil-whisper-large-v3 at 8000 Hz

Date: 2026-11-15. Clips: 128. Language: English.

## Word error rate

The mean word error rate is 0.118.

| Sentence group | Mean word error rate | Clips |
|---|---|---|
| product | 0.127 | 36 |
| order | 0.091 | 24 |
| phone | 0.215 | 16 |
| address | 0.207 | 24 |
| yesno | 0.000 | 28 |

| Voice | Mean word error rate | Clips |
|---|---|---|
| af_heart | 0.109 | 32 |
| am_adam | 0.120 | 32 |
| Samantha | 0.122 | 32 |
| Daniel | 0.123 | 32 |

## Time

Each Whisper call ran alone, after the model loaded and one warm-up call.

- Median time: 0.174 s
- p90 time: 0.183 s

## Worst clips

| Word error rate | Voice | Reference | Heard |
|---|---|---|---|
| 0.43 | Samantha | The tracking number is BD 778 4521. | The tracking number is BD-778-4521. |
| 0.42 | Daniel | Is the ESP32 DevKit better than the Arduino Uno for Wi-Fi projects? | Is the Sp-32 dev kit better than the Arduino for Wi-Fi project? |
| 0.40 | Daniel | Which board has Bluetooth, the ESP32 or the Arduino Uno? | Which board has Bluetooth? The ESP 32 or the Arduino. You know. |
| 0.40 | af_heart | Send it to Agrabad, Chittagong. | Send it to Agribod, Chidegang. |
| 0.40 | am_adam | Send it to Agrabad, Chittagong. | Send it to Agrobod Chittagung. |
| 0.40 | Samantha | Send it to Agrabad, Chittagong. | Send it to a grabout, chittagong. |
| 0.40 | Daniel | Send it to Agrabad, Chittagong. | Send it to the grab-bird. Chittagong. |
| 0.38 | am_adam | Deliver to Gulshan 2, Dhaka, near the circle. | delivered to Gulsion 2, DACA near the circle. |

## Note

Synthetic voices give lower error rates than real callers. Real callers have noise,
accents, and a poor phone line. Use these numbers to compare models, not to predict
the error rate of a real call.
