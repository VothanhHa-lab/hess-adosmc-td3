#!/usr/bin/env python3
"""DO-ASMC baseline (increase-decrease adaptive gain driven by the sliding variable) vs A-DO-SMC.

Tier-1 variants, all with the same NDO, sliding variable, conditional integrator, saturation and sampling:
  a      A-DO-SMC (k_s = k_min + beta*eps, boundary layer phi_s = k_s/k')
  asmc0  DO-ASMC  (dK/dt = gamma*|s|*sgn(|s|-eps_s), K >= K_min, fixed boundary layer phi0), starting values
  asmc   DO-ASMC with eps_s, gamma tuned on the training cycles (scripts/asmc_tune.py)
  afix   A-DO-SMC with the fixed boundary layer phi0 (separates the gain law from the scaled layer)
Cases on US06 (x1.0, x1.1), noise seeds 101-103: constant lambda in {100, 250, 800} 1/s with the fixed supervisor,
and the lambda/tau_b/v_sc,ref schedules of the three filter-v2 agents replayed open-loop (same schedule for all variants).
Seed 101 is logged at 1 ms.  Output: data/processed/asmc_compare.json, data/raw/asmc/*.npz
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from hess.cycles import cycle_profile  # noqa: E402
from hess.env import run_episode  # noqa: E402
from hess.params import CTRL_ADOSMC, CTRL_ASMC  # noqa: E402
from hess.supervisors import FixedSupervisor  # noqa: E402
from lambda_robustness import ReplaySupervisor, schedule  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "nominal.yaml")))
PHI0 = CFG["do_asmc"]["phi0"]["value"]
RAW = os.path.join(ROOT, "data", "raw", "asmc"); os.makedirs(RAW, exist_ok=True)
TUNE = json.load(open(os.path.join(ROOT, "data", "processed", "asmc_tune.json")))
_TK = os.path.join(ROOT, "data", "processed", "asmc_tune_k.json")
TUNEK = json.load(open(_TK)) if os.path.exists(_TK) else None
TIERS = {"a": (CTRL_ADOSMC, {}),
         "asmc0": (CTRL_ASMC, {}),                                                  # starting values (config)
         "asmc": (CTRL_ASMC, {"eps_s": TUNE["eps_s"], "gamma_a": TUNE["gamma_a"]}),  # tuned on the training cycles
         "afix": (CTRL_ADOSMC, {"phi_fix": PHI0})}
if TUNEK:  # DO-ASMC with gain projection K <= K_max and low-pass filtered |s|, tuned on the training cycles
    TIERS["asmck"] = (CTRL_ASMC, {k: TUNEK[k] for k in ("eps_s", "gamma_a", "k_max", "tau_s")})
ITEMS = {"us06": (1.0, 1.1), "wltc_extrahigh_3": (1.0, 1.2)}


def run(args):
    case, tier, item, scale, seed = args
    cfg = copy.deepcopy(CFG)
    if case.startswith("lam"):
        cfg["control"]["lam"]["value"] = float(case[3:])
        sup = FixedSupervisor(cfg)
    else:
        sup = ReplaySupervisor(schedule(int(case[5:]), f"{item}x{scale}"))
    ctrl, ov = TIERS[tier]
    ov = dict(ov); ov["load_scale"] = scale
    le = 20 if seed == 101 else 0
    t0 = time.time()
    ep, s = run_episode(cfg, ctrl, cycle_profile(item, cfg, ROOT), sup, seed=seed, log_every=le, overrides=ov, n_sub=5)
    s["wall_s"] = time.time() - t0
    if le:
        d = ep.log_dict(); m = d["t"] > 0.2
        Dt = np.abs(d["Dtrue"] - d["Dhat"])[m]; r = d["ks"][m] / np.maximum(Dt, 1e-9)
        s.update(Dtilde_max=float(Dt.max()), cond_frac=float(np.mean(r >= 1.0)), cond_minratio=float(r.min()),
                 ks_max=float(d["ks"][m].max()), ks_p99=float(np.percentile(d["ks"][m], 99)),
                 ks_final=float(d["ks"][-1]), phi_max=float(d["phi"][m].max()))
        if scale == 1.0 and item == "us06":
            keep = ("t", "vdc", "ks", "phi", "u", "lam", "Dtrue", "Dhat")
            np.savez_compressed(os.path.join(RAW, f"{case}_{tier}_{item}_x{scale}_s{seed}.npz"),
                                **{k: d[k].astype(np.float32) for k in keep})
    return case, tier, item, scale, seed, s


def main():
    """Usage: asmc_compare.py [items=us06,wltc_extrahigh_3] [tiers=a,asmc0,...]; results are merged into the json."""
    kw = dict(a.split("=") for a in sys.argv[1:])
    items = kw.get("items", "us06").split(","); tiers = kw.get("tiers", ",".join(TIERS)).split(",")
    cases = ["lam100", "lam250", "lam800"] + [f"agent{s}" for s in CFG["agent"]["train_seeds"]]
    jobs = [(c, t, it, sc, sd) for it in items for c in cases for t in tiers for sc in ITEMS[it] for sd in (101, 102, 103)]
    fn = os.path.join(ROOT, "data", "processed", "asmc_compare.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {"results": {}}
    out.update({"tuned": {"eps_s": TUNE["eps_s"], "gamma_a": TUNE["gamma_a"]}, "phi0": PHI0,
                "gamma_a": CFG["do_asmc"]["gamma_a"]["value"], "eps_s": CFG["do_asmc"]["eps_s"]["value"]})
    if TUNEK:
        out["tuned_k"] = {k: TUNEK[k] for k in ("eps_s", "gamma_a", "k_max", "tau_s")}
    with Pool(2) as pool:
        for c, t, it, sc, sd, s in pool.imap_unordered(run, jobs):
            out["results"].setdefault(c, {}).setdefault(t, {}).setdefault(f"{it}x{sc}", {})[str(sd)] = s
            json.dump(out, open(fn, "w"), indent=1, default=float)
            print("[asmc]", c, t, it, sc, sd, round(s["max_dev_V"], 3), round(s["du_rms_A"], 4), s.get("ks_max", ""), flush=True)
    json.dump(out, open(fn, "w"), indent=1, default=float)
    print("[asmc] done")


if __name__ == "__main__":
    main()
