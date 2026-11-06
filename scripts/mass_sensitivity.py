#!/usr/bin/env python3
"""Sensitivity to the SC bank mass, which the main study does not add to the vehicle mass.
Re-evaluates fixed, SOC rule, persistence MPC-EMS and the three filter-v2 agents (trained at
1830 kg, not retrained) on US06 and WLTC high with the vehicle mass increased by the mass of
three BMOD0063 P125 modules (3 x 61 kg, Maxwell datasheet). Noise seed 101 for every run;
the reference values are the seed-101 runs of the main evaluation."""
import json, os, sys
from multiprocessing import Pool
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from hess.cycles import cycle_profile  # noqa: E402
from hess.env import run_episode  # noqa: E402
from hess.params import CTRL_ADOSMC  # noqa: E402
import evaluate as EV  # noqa: E402
ADD_KG = 3 * 61.0
CFGS = [("A-DO-SMC + fixed", "fixed", None), ("A-DO-SMC + rule", "rule", None),
        ("A-DO-SMC + MPC-EMS (persistence)", "mpcp", None)] + \
       [(f"A-DO-SMC + TD3 + filter v2 [s{s}]", "filterv2", f"td3_filterv2_seed{s}") for s in (11, 22, 33)]
ITEMS = ["us06", "wltc_high_3b"]


def job(a):
    name, kind, tag, item = a
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "nominal.yaml")))
    cfg["vehicle"]["mass"]["value"] += ADD_KG
    prof = cycle_profile(item, cfg, ROOT)
    _, s = run_episode(cfg, CTRL_ADOSMC, prof, EV.make_sup(cfg, kind, tag), seed=101,
                       overrides={"load_scale": 1.0}, n_sub=5)
    return name, item, s


if __name__ == "__main__":
    with Pool(2) as p:
        R = p.map(job, [(n, k, t, it) for n, k, t in CFGS for it in ITEMS])
    E = json.load(open(os.path.join(ROOT, "data", "processed", "eval_results.json")))["results"]
    out = {"added_mass_kg": ADD_KG, "runs": {}}
    for name, item, s in R:
        ref = E[name][f"{item}x1.0"]["101"]
        out["runs"][f"{name}|{item}"] = {k: s[k] for k in ("ib_rms_A", "sat_frac", "max_dev_V", "vsc_min_V", "status")}
        out["runs"][f"{name}|{item}"]["ref"] = {k: ref[k] for k in ("ib_rms_A", "sat_frac", "max_dev_V", "vsc_min_V", "status")}
    json.dump(out, open(os.path.join(ROOT, "data", "processed", "mass_sensitivity.json"), "w"), indent=1)
    for k, v in out["runs"].items():
        print(f"{k:60s} ib {v['ref']['ib_rms_A']:6.2f}->{v['ib_rms_A']:6.2f} sat {100*v['ref']['sat_frac']:.2f}->{100*v['sat_frac']:.2f} dev {v['ref']['max_dev_V']:.2f}->{v['max_dev_V']:.2f} vmin {v['vsc_min_V']:.0f} {v['status']}")
