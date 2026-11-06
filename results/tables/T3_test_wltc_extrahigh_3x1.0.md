| Configuration | max|e| (V) | RMS e (V) | RMS i_b (A) | RMS di_b/dt (A/s) | Ah_b | sat. frac (%) | min v_sc (V) | status |
|---|---|---|---|---|---|---|---|---|
| PI + fixed | 2.56 | 0.161 | 60.76 | 92.1 | 4.773 | 0.000 | 218.0 | ok |
| DO-SMC (paper) + fixed | 2.01 | 0.041 | 60.76 | 92.2 | 4.773 | 0.000 | 218.0 | ok |
| Robust MPC + fixed | 1.37 | 0.040 | 60.76 | 92.2 | 4.773 | 0.000 | 218.0 | ok |
| A-DO-SMC + fixed | 1.35 | 0.040 | 60.76 | 92.2 | 4.773 | 0.000 | 218.0 | ok |
| A-DO-SMC + rule | 1.35 | 0.040 | 60.68 | 92.2 | 4.758 | 0.000 | 219.5 | ok |
| A-DO-SMC + TD3 no-filter (mean ± std, 3 train seeds) | 9.74 ± 10.89 | 0.760 ± 1.210 | 55.43 ± 4.20 | 223.6 ± 172.8 | 4.232 ± 0.503 | 0.223 ± 0.387 | 190.4 ± 14.5 | ok,ok,ok |
| A-DO-SMC + TD3 + filter v1 (mean ± std, 3 train seeds) | 1.39 ± 0.18 | 0.050 ± 0.003 | 56.10 ± 1.08 | 139.7 ± 56.7 | 4.253 ± 0.123 | 0.000 ± 0.000 | 192.7 ± 8.8 | ok,ok,ok |
| A-DO-SMC + TD3 + filter v2 (proposed) (mean ± std, 3 train seeds) | 1.29 ± 0.01 | 0.045 ± 0.005 | 56.55 ± 0.87 | 118.9 ± 22.9 | 4.294 ± 0.109 | 0.000 ± 0.000 | 197.1 ± 6.0 | ok,ok,ok |
