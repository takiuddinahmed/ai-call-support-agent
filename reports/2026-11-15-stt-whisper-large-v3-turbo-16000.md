# Speech-to-text benchmark: mlx-community/whisper-large-v3-turbo at 16000 Hz

Date: 2026-11-15. Clips: 128. Language: English.

## Word error rate

The mean word error rate is 0.086.

| Sentence group | Mean word error rate | Clips |
|---|---|---|
| product | 0.053 | 36 |
| order | 0.107 | 24 |
| phone | 0.158 | 16 |
| address | 0.165 | 24 |
| yesno | 0.000 | 28 |

| Voice | Mean word error rate | Clips |
|---|---|---|
| af_heart | 0.095 | 32 |
| am_adam | 0.097 | 32 |
| Samantha | 0.083 | 32 |
| Daniel | 0.067 | 32 |

## Time

Each Whisper call ran alone, after the model loaded and one warm-up call.

- Median time: 0.185 s
- p90 time: 0.196 s

## Worst clips

| Word error rate | Voice | Reference | Heard |
|---|---|---|---|
| 0.43 | af_heart | The tracking number is BD 778 4521. | The tracking number is BD778-4521. |
| 0.43 | am_adam | The tracking number is BD 778 4521. | The tracking number is BD778-4521. |
| 0.43 | Samantha | The tracking number is BD 778 4521. | The tracking number is BD7784521. |
| 0.43 | Daniel | The tracking number is BD 778 4521. | The tracking number is BD7784521. |
| 0.40 | af_heart | Send it to Agrabad, Chittagong. | Send it to Agrabod, Chiragang. |
| 0.40 | am_adam | Send it to Agrabad, Chittagong. | Send it to Agrabahd, Chiragung. |
| 0.40 | Samantha | Send it to Agrabad, Chittagong. | Send it to a grabout. Chittagong. |
| 0.40 | Daniel | Send it to Agrabad, Chittagong. | Send it to a grabberd, Chittagong. |

## Note

Synthetic voices give lower error rates than real callers. Real callers have noise,
accents, and a poor phone line. Use these numbers to compare models, not to predict
the error rate of a real call.
