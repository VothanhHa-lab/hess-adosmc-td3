#!/usr/bin/env python3
"""Tier-1 study on the Tesla Model 3 plant (fixed supervisor).

(1) Worst-case sizing of a high-gain fixed DO-SMC from the TRAINING cycles only:
    k_s,hi = 1.1 * max|D_tilde| observed with A-DO-SMC on the four training cycles
    (1 ms logging, t > 0.2 s), rounded up to 100 V/s; phi_s = k_s,hi / k'.
(2) Held-out test items x noise seeds 101-103 for the high-gain DO-SMC.
(3) 1 ms logs of US06 (seed 101) for DO-SMC (low gain), DO-SMC (high gain), robust MPC, A-DO-SMC,
    and a noise-free A-DO-SMC run, for Fig. 3 and the estimator-condition check.
Output: data/processed/tier1_m3.json, data/raw/tier1_m3/*.npz
"""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hess.cycles import cycle_profile  # noqa: E402
from hess.env import run_episode  # noqa: E402
from hess.params import CTRL_ADOSMC, CTRL_DOSMC, CTRL_MPC  # noqa: E402
from hess.supervisors import FixedSupervisor  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "nominal.yaml")))
RAW = os.path.join(ROOT, "data", "raw", "tier1_m3"); os.makedirs(RAW, exist_ok=True)
KPRIME = CFG["control"]["ks"]["value"] / CFG["control"]["phi"]["value"]


def run(args):
    tag, ctrl, item, scale, seed, ov, le = args
    prof = cycle_profile(item, CFG, ROOT)
    o = {"load_scale": scale}; o.update(ov)
    t0 = time.time()
    ep, s = run_episode(CFG, ctrl, prof, FixedSupervisor(CFG), seed=seed, log_every=le, overrides=o, n_sub=5)
    s["wall_s"] = time.time() - t0
    if le:
        d = ep.log_dict(); m = d["t"] > 0.2
        Dt = np.abs(d["Dtrue"] - d["Dhat"])[m]
        s["Dtilde_max"] = float(Dt.max())
        if ctrl in (CTRL_ADOSMC, CTRL_DOSMC):
            r = d["ks"][m] / np.maximum(Dt, 1e-9)
            s["cond_frac"] = float(np.mean(r >= 1.0)); s["cond_minratio"] = float(r.min())
        np.savez_compressed(os.path.join(RAW, f"{tag}_{item}_x{scale}_s{seed}.npz"),
                            **{k: v.astype(np.float32) for k, v in d.items()})
    return tag, item, scale, seed, s


def main():
    out = {"kprime": KPRIME}
    with Pool(2) as pool:
        # (1) training cycles
        tr = [(f"adosmc_train", CTRL_ADOSMC, c, 1.0, 101, {}, 20) for c in CFG["cycles"]["train"]]
        R = pool.map(run, tr)
        dmax = max(r[4]["Dtilde_max"] for r in R)
        khi = 100.0 * math.ceil(1.1 * dmax / 100.0)
        out["train_Dtilde_max"] = {r[1]: r[4]["Dtilde_max"] for r in R}
        out["ks_hi"] = khi; out["phi_hi"] = khi / KPRIME
        print("[tier1] ks_hi =", khi, flush=True)
        hi = {"ks": khi, "phi": khi / KPRIME}
        # (2) test items, high-gain DO-SMC
        jobs = [("dosmc_hi", CTRL_DOSMC, c, float(sc), sd, hi, 0)
                for c, sc in CFG["agent"]["test_set"] for sd in (101, 102, 103)]
        # (3) fine logs on US06
        jobs += [("dosmc", CTRL_DOSMC, "us06", 1.0, 101, {}, 20),
                 ("dosmc_hi_log", CTRL_DOSMC, "us06", 1.0, 101, hi, 20),
                 ("mpc", CTRL_MPC, "us06", 1.0, 101, {}, 20),
                 ("adosmc", CTRL_ADOSMC, "us06", 1.0, 101, {}, 20),
                 ("adosmc_nonoise", CTRL_ADOSMC, "us06", 1.0, 101, {"sig_v": 0.0, "sig_i": 0.0}, 20),
                 ("dosmc_nonoise", CTRL_DOSMC, "us06", 1.0, 101, {"sig_v": 0.0, "sig_i": 0.0}, 20)]
        res = {}
        for tag, it, sc, sd, s in pool.imap_unordered(run, jobs):
            res.setdefault(tag, {}).setdefault(f"{it}x{sc}", {})[str(sd)] = s
            print("[tier1]", tag, it, sc, sd, round(s["max_dev_V"], 3), flush=True)
    out["results"] = res
    json.dump(out, open(os.path.join(ROOT, "data", "processed", "tier1_m3.json"), "w"), indent=1, default=float)
    print("[tier1] done")


if __name__ == "__main__":
    main()
