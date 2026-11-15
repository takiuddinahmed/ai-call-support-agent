# Speech-to-text benchmark: mlx-community/whisper-large-v3-turbo at 8000 Hz

Date: 2026-11-15. Clips: 128. Language: English.

## Word error rate

The mean word error rate is 0.082.

| Sentence group | Mean word error rate | Clips |
|---|---|---|
| product | 0.053 | 36 |
| order | 0.108 | 24 |
| phone | 0.146 | 16 |
| address | 0.154 | 24 |
| yesno | 0.000 | 28 |

| Voice | Mean word error rate | Clips |
|---|---|---|
| af_heart | 0.080 | 32 |
| am_adam | 0.102 | 32 |
| Samantha | 0.074 | 32 |
| Daniel | 0.073 | 32 |

## Time

Each Whisper call ran alone, after the model loaded and one warm-up call.

- Median time: 0.186 s
- p90 time: 0.194 s

## Worst clips

| Word error rate | Voice | Reference | Heard |
|---|---|---|---|
| 0.60 | Daniel | Send it to Agrabad, Chittagong. | Send it to the grabberd, Chittergong. |
| 0.43 | am_adam | The tracking number is BD 778 4521. | The tracking number is BD778-4521. |
| 0.43 | Samantha | The tracking number is BD 778 4521. | The tracking number is BD7784521. |
| 0.43 | Daniel | The tracking number is BD 778 4521. | The tracking number is BD-778-4521. |
| 0.43 | Samantha | Please send the message to 01811 556677. | please send a message to 01811-556677. |
| 0.40 | af_heart | Send it to Agrabad, Chittagong. | Send it to Agrabahd, Chittagang. |
| 0.40 | am_adam | Send it to Agrabad, Chittagong. | Send it to Agrabahd, Chiragung. |
| 0.40 | Samantha | Send it to Agrabad, Chittagong. | Send it to a grabout. Chittagong. |

## Note

Synthetic voices give lower error rates than real callers. Real callers have noise,
accents, and a poor phone line. Use these numbers to compare models, not to predict
the error rate of a real call.
