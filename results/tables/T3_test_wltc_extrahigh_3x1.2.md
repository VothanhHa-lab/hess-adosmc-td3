| Configuration | max|e| (V) | RMS e (V) | RMS i_b (A) | RMS di_b/dt (A/s) | Ah_b | sat. frac (%) | min v_sc (V) | status |
|---|---|---|---|---|---|---|---|---|
| PI + fixed | 3.07 | 0.192 | 62.83 | 88.4 | 5.009 | 0.000 | 179.1 | ok |
| DO-SMC (paper) + fixed | 2.49 | 0.045 | 62.83 | 88.5 | 5.009 | 0.000 | 179.1 | ok |
| Robust MPC + fixed | 1.65 | 0.043 | 62.83 | 88.5 | 5.009 | 0.000 | 179.1 | ok |
| A-DO-SMC + fixed | 1.62 | 0.043 | 62.83 | 88.5 | 5.009 | 0.000 | 179.1 | ok |
| A-DO-SMC + rule | 1.62 | 0.044 | 62.56 | 90.6 | 4.991 | 0.000 | 181.0 | ok |
| A-DO-SMC + TD3 no-filter (mean ± std, 3 train seeds) | 75.75 ± 107.77 | 11.097 ± 15.300 | 62.29 ± 1.73 | 1396.5 ± 2169.9 | 4.455 ± 0.719 | 3.524 ± 3.074 | 160.1 ± 8.8 | 3 failed,ok,ok |
| A-DO-SMC + TD3 + filter v1 (mean ± std, 3 train seeds) | 60.48 ± 79.91 | 4.691 ± 6.824 | 59.21 ± 1.13 | 462.0 ± 622.7 | 4.593 ± 0.142 | 0.482 ± 0.595 | 156.1 ± 5.2 | ok,ok,ok |
| A-DO-SMC + TD3 + filter v2 (proposed) (mean ± std, 3 train seeds) | 1.54 ± 0.00 | 0.052 ± 0.008 | 60.62 ± 0.86 | 98.8 ± 19.0 | 4.749 ± 0.083 | 0.000 ± 0.000 | 165.0 ± 5.5 | ok,ok,ok |
