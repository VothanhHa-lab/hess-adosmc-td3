#!/usr/bin/env python3
"""Validation return/step of the non-learning supervisors (same protocol as checkpoint selection)."""
import json, os, sys
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hess.cycles import cycle_profile  # noqa: E402
from hess.env import run_episode  # noqa: E402
from hess.params import CTRL_ADOSMC  # noqa: E402
from hess.supervisors import FixedSupervisor, RuleSupervisor  # noqa: E402
cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "nominal.yaml")))
out = {}
for sup in (FixedSupervisor(cfg), RuleSupervisor(cfg)):
    r = []
    for n, sc in cfg["agent"]["validation_set"]:
        _, s = run_episode(cfg, CTRL_ADOSMC, cycle_profile(n, cfg, ROOT), sup, seed=777, overrides={"load_scale": sc})
        r.append(s["return"] / s["n_decisions"])
    out[sup.name] = float(np.mean(r))
json.dump(out, open(os.path.join(ROOT, "data", "processed", "validation_baselines.json"), "w"), indent=1)
print(out)
