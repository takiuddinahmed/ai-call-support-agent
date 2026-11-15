# Speech-to-text benchmark: mlx-community/distil-whisper-large-v3 at 16000 Hz

Date: 2026-11-15. Clips: 128. Language: English.

## Word error rate

The mean word error rate is 0.112.

| Sentence group | Mean word error rate | Clips |
|---|---|---|
| product | 0.119 | 36 |
| order | 0.091 | 24 |
| phone | 0.203 | 16 |
| address | 0.190 | 24 |
| yesno | 0.000 | 28 |

| Voice | Mean word error rate | Clips |
|---|---|---|
| af_heart | 0.105 | 32 |
| am_adam | 0.120 | 32 |
| Samantha | 0.113 | 32 |
| Daniel | 0.109 | 32 |

## Time

Each Whisper call ran alone, after the model loaded and one warm-up call.

- Median time: 0.176 s
- p90 time: 0.185 s

## Worst clips

| Word error rate | Voice | Reference | Heard |
|---|---|---|---|
| 0.43 | Samantha | The tracking number is BD 778 4521. | The tracking number is BD-778-4521. |
| 0.42 | Daniel | Is the ESP32 DevKit better than the Arduino Uno for Wi-Fi projects? | Is the Sp 32 dev kit better than the Arduino for Wi-Fi projects? |
| 0.40 | Daniel | Which board has Bluetooth, the ESP32 or the Arduino Uno? | Which board has Bluetooth, the Sp 32 or the Arduino, you know. |
| 0.40 | af_heart | Send it to Agrabad, Chittagong. | Send it to Agribod, Chidegang. |
| 0.40 | am_adam | Send it to Agrabad, Chittagong. | Send it to Agrobod Chidagung. |
| 0.40 | Samantha | Send it to Agrabad, Chittagong. | Send it to a grabout, chittagong. |
| 0.40 | Daniel | Send it to Agrabad, Chittagong. | Send it to a grab-od. Chittagong. |
| 0.38 | am_adam | Deliver to Gulshan 2, Dhaka, near the circle. | delivered to Gulsion 2, DACA near the circle. |

## Note

Synthetic voices give lower error rates than real callers. Real callers have noise,
accents, and a poor phone line. Use these numbers to compare models, not to predict
the error rate of a real call.
