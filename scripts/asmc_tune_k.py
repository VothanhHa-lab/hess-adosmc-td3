#!/usr/bin/env python3
"""Tune the DO-ASMC variant with gain projection and a filtered sliding variable on the TRAINING cycles only
(fixed supervisor, nominal lambda, noise seed 101), with the same grid and selection rule as asmc_tune.py.
Variant: K_min = K(0) = 120 V/s <= K <= K_max = k_s,hi (the worst-case bound of the high-gain DO-SMC, from
data/processed/tier1_m3.json), |s| in the adaptation law low-pass filtered with tau_s = tau_f, fixed layer phi0.
Output: data/processed/asmc_tune_k.json"""
import json, os, sys, itertools
from multiprocessing import Pool
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hess.cycles import cycle_profile  # noqa: E402
from hess.env import run_episode  # noqa: E402
from hess.params import CTRL_ASMC  # noqa: E402
from hess.supervisors import FixedSupervisor  # noqa: E402
CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "nominal.yaml")))
T1 = json.load(open(os.path.join(ROOT, "data", "processed", "tier1_m3.json")))
KMAX = float(T1["ks_hi"]); TAUS = float(CFG["supervision"]["tau_f"]["value"])
EPS = [0.25, 0.5, 1.0, 2.0]; GAM = [1.0e3, 1.0e4]

def run(a):
    eps, gam, cyc = a
    ep, s = run_episode(CFG, CTRL_ASMC, cycle_profile(cyc, CFG, ROOT), FixedSupervisor(CFG), seed=101,
                        overrides={"eps_s": eps, "gamma_a": gam, "k_max": KMAX, "tau_s": TAUS}, n_sub=5)
    return eps, gam, cyc, s

if __name__ == "__main__":
    jobs = [(e, g, c) for e, g in itertools.product(EPS, GAM) for c in CFG["cycles"]["train"]]
    res = {}
    with Pool(2) as pool:
        for e, g, c, s in pool.imap_unordered(run, jobs):
            res.setdefault(f"{e}|{g}", {})[c] = s
            print("[tuneK]", e, g, c, round(s["max_dev_V"], 3), round(s["sat_frac"], 4), s["status"], flush=True)
    score = {}
    for k, v in res.items():
        ok = all(x["status"] == "ok" and x["sat_frac"] < 0.01 for x in v.values())
        score[k] = (max(x["max_dev_V"] for x in v.values()) if ok else float("inf"),
                    max(x["du_rms_A"] for x in v.values()))
    best = min(score, key=lambda k: score[k])
    out = {"grid": res, "score": {k: list(v) for k, v in score.items()}, "best": best, "k_max": KMAX, "tau_s": TAUS,
           "eps_s": float(best.split("|")[0]), "gamma_a": float(best.split("|")[1])}
    json.dump(out, open(os.path.join(ROOT, "data", "processed", "asmc_tune_k.json"), "w"), indent=1, default=float)
    print("[tuneK] best", best, score[best])
