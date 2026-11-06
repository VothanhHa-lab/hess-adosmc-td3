#!/usr/bin/env python3
"""Robustness of the tier-1 reaching condition to the observer gain lambda (Tesla Model 3 plant).

Compares the fixed high-gain DO-SMC (k_s = k_s,hi from data/processed/tier1_m3.json, sized at the nominal
lambda = 250 1/s) with A-DO-SMC on US06 (x1.0, x1.1), noise seeds 101-103:
 (A) constant lambda in {100, 250, 800} 1/s (filter bounds and nominal), fixed tau_b and v_sc,ref;
 (B) the supervisory schedule theta(t) = (tau_b, v_sc,ref, lambda) chosen by each of the three filter-v2
     TD3 agents in their seed-101 evaluation run on the same cycle, replayed open-loop, so that both
     tier-1 controllers see exactly the same lambda trajectory.
Seed 101 is logged at 1 ms to evaluate k_s >= |D_tilde| (t > 0.2 s).
Output: data/processed/lambda_robustness.json
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hess.cycles import cycle_profile  # noqa: E402
from hess.env import run_episode  # noqa: E402
from hess.params import CTRL_ADOSMC, CTRL_DOSMC  # noqa: E402
from hess.supervisors import FixedSupervisor  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "nominal.yaml")))
T1 = json.load(open(os.path.join(ROOT, "data", "processed", "tier1_m3.json")))
HI = {"ks": T1["ks_hi"], "phi": T1["phi_hi"]}
E = json.load(open(os.path.join(ROOT, "data", "processed", "eval_results.json")))["results"]
RAWE = os.path.join(ROOT, "data", "raw", "eval")


class ReplaySupervisor:
    name = "replay"

    def __init__(self, decisions):
        self.D = np.asarray(decisions, float)[:, 1:4]
        self.k = 0

    def reset(self):
        self.k = 0

    def decide(self, obs, info):
        th = self.D[min(self.k, len(self.D) - 1)].copy()
        self.k += 1
        return th, None


def schedule(seed_tr, item):
    n = f"A-DO-SMC + TD3 + filter v2 [s{seed_tr}]"
    return np.load(os.path.join(RAWE, E[n][item]["101"]["raw"] + ".npz"))["decisions"]


def run(args):
    case, tier, item, scale, seed = args
    cfg = copy.deepcopy(CFG)
    if case.startswith("lam"):
        cfg["control"]["lam"]["value"] = float(case[3:])
        sup = FixedSupervisor(cfg)
    else:
        sup = ReplaySupervisor(schedule(int(case[5:]), f"{item}x{scale}"))
    ctrl, ov = (CTRL_DOSMC, dict(HI)) if tier == "hi" else (CTRL_ADOSMC, {})
    ov["load_scale"] = scale
    le = 20 if seed == 101 else 0
    t0 = time.time()
    ep, s = run_episode(cfg, ctrl, cycle_profile(item, cfg, ROOT), sup, seed=seed, log_every=le, overrides=ov, n_sub=5)
    s["wall_s"] = time.time() - t0
    if le:
        d = ep.log_dict(); m = d["t"] > 0.2
        Dt = np.abs(d["Dtrue"] - d["Dhat"])[m]; r = d["ks"][m] / np.maximum(Dt, 1e-9)
        s.update(Dtilde_max=float(Dt.max()), cond_frac=float(np.mean(r >= 1.0)), cond_minratio=float(r.min()),
                 lam_min=float(d["lam"][m].min()), lam_max=float(d["lam"][m].max()),
                 lam_mean=float(d["lam"][m].mean()))
    return case, tier, item, scale, seed, s


def main():
    cases = ["lam100", "lam250", "lam800"] + [f"agent{s}" for s in CFG["agent"]["train_seeds"]]
    jobs = [(c, t, "us06", sc, sd) for c in cases for t in ("hi", "a") for sc in (1.0, 1.1) for sd in (101, 102, 103)]
    out = {"ks_hi": T1["ks_hi"], "phi_hi": T1["phi_hi"], "results": {}}
    with Pool(2) as pool:
        for c, t, it, sc, sd, s in pool.imap_unordered(run, jobs):
            out["results"].setdefault(c, {}).setdefault(t, {}).setdefault(f"{it}x{sc}", {})[str(sd)] = s
            print("[lam]", c, t, it, sc, sd, round(s["max_dev_V"], 3), s.get("cond_minratio", ""), flush=True)
    json.dump(out, open(os.path.join(ROOT, "data", "processed", "lambda_robustness.json"), "w"), indent=1, default=float)
    print("[lam] done")


if __name__ == "__main__":
    main()
