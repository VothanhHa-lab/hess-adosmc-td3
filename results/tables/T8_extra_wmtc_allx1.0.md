| Configuration | max|e| (V) | RMS e (V) | RMS i_b (A) | RMS di_b/dt (A/s) | Ah_b | sat. frac (%) | min v_sc (V) | status |
|---|---|---|---|---|---|---|---|---|
| DO-SMC (paper) + fixed | 2.13 | 0.071 | 48.26 | 135.5 | 18.919 | 0.000 | 214.8 | ok |
| Robust MPC + fixed | 1.33 | 0.040 | 48.26 | 135.5 | 18.919 | 0.000 | 214.8 | ok |
| A-DO-SMC + fixed | 1.32 | 0.040 | 48.26 | 135.5 | 18.919 | 0.000 | 214.8 | ok |
| A-DO-SMC + rule | 1.32 | 0.041 | 46.49 | 137.8 | 18.051 | 0.000 | 219.3 | ok |
| A-DO-SMC + TD3 + filter v1 (mean ± std, 3 train seeds) | 77.73 ± 65.69 | 2.238 ± 2.151 | 45.33 ± 2.12 | 161.3 ± 52.4 | 17.821 ± 0.942 | 0.101 ± 0.098 | 174.4 ± 20.9 | ok,ok,ok |
| A-DO-SMC + TD3 + filter v2 (proposed) (mean ± std, 3 train seeds) | 1.41 ± 0.04 | 0.044 ± 0.002 | 45.34 ± 0.47 | 148.2 ± 22.8 | 17.649 ± 0.183 | 0.000 ± 0.000 | 189.5 ± 8.8 | ok,ok,ok |
