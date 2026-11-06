| Configuration | max|e| (V) | RMS e (V) | RMS i_b (A) | RMS di_b/dt (A/s) | Ah_b | sat. frac (%) | min v_sc (V) | status |
|---|---|---|---|---|---|---|---|---|
| DO-SMC (paper) + fixed | 1.97 | 0.061 | 24.87 | 154.6 | 9.323 | 0.000 | 248.9 | ok |
| Robust MPC + fixed | 1.33 | 0.039 | 24.87 | 154.6 | 9.323 | 0.000 | 248.9 | ok |
| A-DO-SMC + fixed | 1.32 | 0.038 | 24.87 | 154.6 | 9.323 | 0.000 | 248.9 | ok |
| A-DO-SMC + rule | 1.32 | 0.039 | 22.06 | 154.6 | 8.639 | 0.000 | 247.7 | ok |
| A-DO-SMC + TD3 + filter v1 (mean ± std, 3 train seeds) | 1.31 ± 0.06 | 0.041 ± 0.003 | 22.79 ± 1.95 | 135.5 ± 12.1 | 9.271 ± 0.683 | 0.000 ± 0.000 | 221.0 ± 14.3 | ok,ok,ok |
| A-DO-SMC + TD3 + filter v2 (proposed) (mean ± std, 3 train seeds) | 1.38 ± 0.22 | 0.041 ± 0.002 | 20.66 ± 0.08 | 153.1 ± 31.5 | 8.688 ± 0.020 | 0.000 ± 0.000 | 227.3 ± 7.8 | ok,ok,ok |
