#!/usr/bin/env python3
"""Train-set statistic used by safety filter v2: largest contiguous energy deficit
E = max over episodes of  integral (P_L - vb*Ib_max)+ dt  on the TRAIN cycles only."""
import os, sys
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hess.cycles import cycle_profile  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "nominal.yaml")))
Pb = cfg["plant"]["ocv_nom"]["value"] * cfg["limits"]["Ibmax"]["value"]
best = (0.0, None)
for n in cfg["cycles"]["train"]:
    c = cycle_profile(n, cfg, ROOT)
    acc = 0.0
    for p in c["P"]:
        acc = acc + (p - Pb) * c["dt"] if p > Pb else 0.0
        if acc > best[0]:
            best = (acc, n)
print(f"P_b,max = {Pb:.0f} W; largest contiguous deficit = {best[0]:.0f} J on '{best[1]}'")
