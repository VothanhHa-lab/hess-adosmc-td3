| Configuration | max|e| (V) | RMS e (V) | RMS i_b (A) | RMS di_b/dt (A/s) | Ah_b | sat. frac (%) | min v_sc (V) | status |
|---|---|---|---|---|---|---|---|---|
| DO-SMC (paper) + fixed | 1.99 | 0.036 | 5.19 | 153.1 | 1.354 | 0.000 | 249.9 | ok |
| Robust MPC + fixed | 1.34 | 0.036 | 5.19 | 153.1 | 1.354 | 0.000 | 249.9 | ok |
| A-DO-SMC + fixed | 1.32 | 0.036 | 5.19 | 153.1 | 1.354 | 0.000 | 249.9 | ok |
| A-DO-SMC + rule | 1.32 | 0.036 | 4.98 | 153.0 | 1.338 | 0.000 | 249.3 | ok |
| A-DO-SMC + TD3 + filter v1 (mean ± std, 3 train seeds) | 1.26 ± 0.00 | 0.039 ± 0.003 | 14.67 ± 2.41 | 253.4 ± 274.9 | 4.460 ± 1.078 | 0.000 ± 0.000 | 222.3 ± 16.6 | ok,ok,ok |
| A-DO-SMC + TD3 + filter v2 (proposed) (mean ± std, 3 train seeds) | 1.26 ± 0.00 | 0.039 ± 0.001 | 16.30 ± 0.91 | 202.7 ± 169.0 | 5.149 ± 0.456 | 0.000 ± 0.000 | 228.8 ± 5.8 | ok,ok,ok |
