# First pilot, re-read against the correct null model

| Model | T | Sink mass (mean ± sd, 3 seeds) | 1/T | Ratio to 1/T | Uniform causal reference | Ratio to reference | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Softmax | 96 | 0.316 ± 0.198 | 0.0104 | 30.4 | 0.0437 | 7.2 | 32.0 |
| Softmax | 192 | 0.239 ± 0.154 | 0.0052 | 45.8 | 0.0253 | 9.4 | 27.1 |
| Softmax | 384 | 0.183 ± 0.122 | 0.0026 | 70.2 | 0.0144 | 12.7 | 34.9 |
| Softmax | 768 | 0.126 ± 0.090 | 0.0013 | 97.0 | 0.0081 | 15.6 | 26.3 |
| Softmax + gate | 96 | 0.261 ± 0.175 | 0.0104 | 25.0 | 0.0437 | 6.0 | 28.1 |
| Softmax + gate | 192 | 0.211 ± 0.152 | 0.0052 | 40.4 | 0.0253 | 8.3 | 29.7 |
| Softmax + gate | 384 | 0.160 ± 0.116 | 0.0026 | 61.4 | 0.0144 | 11.1 | 30.5 |
| Softmax + gate | 768 | 0.103 ± 0.085 | 0.0013 | 79.4 | 0.0081 | 12.8 | 31.8 |
| Hybrid 3:1 | 96 | 0.183 ± 0.103 | 0.0104 | 17.6 | 0.0437 | 4.2 | 6.0 |
| Hybrid 3:1 | 192 | 0.132 ± 0.070 | 0.0052 | 25.3 | 0.0253 | 5.2 | 8.1 |
| Hybrid 3:1 | 384 | 0.101 ± 0.055 | 0.0026 | 38.9 | 0.0144 | 7.0 | 9.9 |
| Hybrid 3:1 | 768 | 0.075 ± 0.042 | 0.0013 | 57.9 | 0.0081 | 9.3 | 7.3 |
| Hybrid + AttnRes | 96 | 0.202 ± 0.180 | 0.0104 | 19.4 | 0.0437 | 4.6 | 5.7 |
| Hybrid + AttnRes | 192 | 0.131 ± 0.094 | 0.0052 | 25.2 | 0.0253 | 5.2 | 8.9 |
| Hybrid + AttnRes | 384 | 0.079 ± 0.048 | 0.0026 | 30.4 | 0.0144 | 5.5 | 9.1 |
| Hybrid + AttnRes | 768 | 0.046 ± 0.024 | 0.0013 | 35.6 | 0.0081 | 5.7 | 7.0 |
| Softmax, answer-only loss | 96 | 0.049 ± 0.003 | 0.0104 | 4.7 | 0.0437 | 1.1 | 33.3 |
| Softmax, answer-only loss | 192 | 0.026 ± 0.001 | 0.0052 | 5.1 | 0.0253 | 1.0 | 29.4 |
| Softmax, answer-only loss | 384 | 0.016 ± 0.000 | 0.0026 | 6.0 | 0.0144 | 1.1 | 31.2 |
| Softmax, answer-only loss | 768 | 0.010 ± 0.001 | 0.0013 | 7.7 | 0.0081 | 1.2 | 18.8 |

## Claims in the first version, recomputed

- "roughly thirty times an even split" at T = 96: 30.4x against 1/T, 7.2x against the uniform causal reference.
- "about ninety seven times an even share" at T = 768: 97.0x against 1/T, 15.6x against the reference.
- The answer-only control measured 0.049 at T = 96, and uniform causal attention gives 0.044 (1.11x). Positions outside the loss receive no gradient, so this control shows attention that was never trained, not a sink removed by the objective.
