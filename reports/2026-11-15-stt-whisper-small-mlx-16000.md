# Speech-to-text benchmark: mlx-community/whisper-small-mlx at 16000 Hz

Date: 2026-11-15. Clips: 128. Language: English.

## Word error rate

The mean word error rate is 0.080.

| Sentence group | Mean word error rate | Clips |
|---|---|---|
| product | 0.043 | 36 |
| order | 0.127 | 24 |
| phone | 0.083 | 16 |
| address | 0.181 | 24 |
| yesno | 0.000 | 28 |

| Voice | Mean word error rate | Clips |
|---|---|---|
| af_heart | 0.072 | 32 |
| am_adam | 0.089 | 32 |
| Samantha | 0.082 | 32 |
| Daniel | 0.078 | 32 |

## Time

Each Whisper call ran alone, after the model loaded and one warm-up call.

- Median time: 0.067 s
- p90 time: 0.082 s

## Worst clips

| Word error rate | Voice | Reference | Heard |
|---|---|---|---|
| 0.60 | Samantha | Send it to Agrabad, Chittagong. | Send it to a grab at Chittagong. |
| 0.43 | am_adam | The tracking number is BD 778 4521. | The tracking number is BD7784521. |
| 0.43 | Samantha | The tracking number is BD 778 4521. | The tracking number is BD7784521. |
| 0.43 | Daniel | The tracking number is BD 778 4521. | The tracking number is BD7784521. |
| 0.40 | af_heart | Send it to Agrabad, Chittagong. | Send it to Agrabod, Chidegang. |
| 0.40 | am_adam | Send it to Agrabad, Chittagong. | Send it to Agrabah Chidagung. |
| 0.40 | Daniel | Send it to Agrabad, Chittagong. | Send it to a grabbered, Chittagong. |
| 0.36 | Samantha | My order number is 3015 and my email is rahim at example dot com. | My order number is 3015 and my email is raheematexample.com |

## Note

Synthetic voices give lower error rates than real callers. Real callers have noise,
accents, and a poor phone line. Use these numbers to compare models, not to predict
the error rate of a real call.
