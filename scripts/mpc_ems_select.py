#!/usr/bin/env python3
"""Horizon selection for the MPC-EMS baseline on the VALIDATION set only.

Same protocol as TD3 checkpoint selection and scripts/validation_baselines.py
(validation items of cfg.agent.validation_set, noise seed 777, metric = mean return per
decision).  Candidates: horizon in {2, 5, 10} s, for each forecast model.  Also re-runs the
fixed supervisor as a regression check of the simulator after adding the P* statistic.
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hess.cycles import cycle_profile  # noqa: E402
from hess.env import run_episode  # noqa: E402
from hess.params import CTRL_ADOSMC  # noqa: E402
from hess.supervisors import FixedSupervisor, MPCEMSSupervisor  # noqa: E402
CFG = os.path.join(ROOT, "configs", "nominal.yaml")
HORIZONS = [2.0, 5.0, 10.0]


def job(a):
    fc, hs, item, sc = a
    cfg = yaml.safe_load(open(CFG))
    sup = FixedSupervisor(cfg) if fc == "fixed" else MPCEMSSupervisor(cfg, fc, hs)
    t0 = time.time()
    _, s = run_episode(cfg, CTRL_ADOSMC, cycle_profile(item, cfg, ROOT), sup, seed=777,
                       overrides={"load_scale": sc})
    st = float(np.median(sup.solve_times)) if fc != "fixed" else None
    return fc, hs, item, sc, s["return"] / s["n_decisions"], s["status"], st, time.time() - t0


if __name__ == "__main__":
    cfg = yaml.safe_load(open(CFG))
    V = [(n, float(sc)) for n, sc in cfg["agent"]["validation_set"]]
    jobs = [("fixed", 0.0, n, sc) for n, sc in V]
    jobs += [(fc, hs, n, sc) for fc in ("persistence", "preview") for hs in HORIZONS for n, sc in V]
    with Pool(2) as p:
        R = p.map(job, jobs)
    out = {}
    for fc, hs, item, sc, r, stt, solve, w in R:
        d = out.setdefault(fc, {}).setdefault(str(hs), {"items": {}})
        d["items"][f"{item}x{sc}"] = {"return_per_step": r, "status": stt, "solve_median_s": solve, "wall_s": w}
    for fc in out:
        for hs, d in out[fc].items():
            d["mean_return_per_step"] = float(np.mean([v["return_per_step"] for v in d["items"].values()]))
    sel = {fc: max(out[fc], key=lambda h: out[fc][h]["mean_return_per_step"]) for fc in ("persistence", "preview")}
    out["selected_horizon_s"] = sel
    json.dump(out, open(os.path.join(ROOT, "data", "processed", "mpc_ems_selection.json"), "w"), indent=1)
    for fc in ("fixed", "persistence", "preview"):
        for hs, d in out[fc].items():
            print(fc, hs, round(d["mean_return_per_step"], 4), [v["status"] for v in d["items"].values()])
    print("selected", sel)
