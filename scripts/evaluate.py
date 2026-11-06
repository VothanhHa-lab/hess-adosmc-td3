#!/usr/bin/env python3
"""Held-out evaluation of every controller/supervisor configuration.

Test items (cycle, load scale) come from cfg.agent.test_set and were never used for
training or checkpoint selection.  Every configuration sees the same items and the
same measurement-noise seeds (paired design).

    python scripts/evaluate.py            # all configs, writes data/processed/eval_*.json
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hess.cycles import cycle_profile, scenario_profile  # noqa: E402
from hess.env import run_episode  # noqa: E402
from hess.params import CTRL_ADOSMC, CTRL_DOSMC, CTRL_MPC, CTRL_PI  # noqa: E402
from hess.scenarios import transient, urban  # noqa: E402
from hess.supervisors import AgentSupervisor, FixedSupervisor, MPCEMSSupervisor, RuleSupervisor  # noqa: E402
from hess.td3 import MLP  # noqa: E402

CFG = os.path.join(ROOT, "configs", "nominal.yaml")
AG = os.path.join(ROOT, "data", "agents")
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw", "eval")
NOISE_SEEDS = [101, 102, 103]
LOG_SEED = 101          # decimated time series stored for this seed (figures)
LOG_EVERY = 200         # 10 ms


def load_actor(tag, cfg):
    H = cfg["agent"]["td3"]["hidden"]
    m = MLP([11, H, H, 3], np.random.default_rng(0), "tanh")
    z = np.load(os.path.join(AG, tag + "_best.npz"))
    m.load([z[f"arr_{i}"] for i in range(len(z.files))])
    return m


def configs(cfg):
    C = [("PI + fixed", CTRL_PI, "fixed", None),
         ("DO-SMC (paper) + fixed", CTRL_DOSMC, "fixed", None),
         ("Robust MPC + fixed", CTRL_MPC, "fixed", None),
         ("A-DO-SMC + fixed", CTRL_ADOSMC, "fixed", None),
         ("A-DO-SMC + rule", CTRL_ADOSMC, "rule", None),
         ("A-DO-SMC + MPC-EMS (persistence)", CTRL_ADOSMC, "mpcp", None),
         ("A-DO-SMC + MPC-EMS (preview)", CTRL_ADOSMC, "mpcv", None)]
    for s in cfg["agent"]["train_seeds"]:
        C.append((f"A-DO-SMC + TD3 no-filter [s{s}]", CTRL_ADOSMC, "nofilter", f"td3_nofilter_seed{s}"))
        C.append((f"A-DO-SMC + TD3 + filter [s{s}]", CTRL_ADOSMC, "filter", f"td3_filter_seed{s}"))
    for s in cfg["agent"]["train_seeds"]:
        C.append((f"A-DO-SMC + TD3 + filter v2 [s{s}]", CTRL_ADOSMC, "filterv2", f"td3_filterv2_seed{s}"))
    return C


EXTRA_CONFIGS = ("A-DO-SMC + fixed", "A-DO-SMC + rule", "A-DO-SMC + MPC-EMS", "A-DO-SMC + TD3 + filter [", "A-DO-SMC + TD3 + filter v2 [",
                 "Robust MPC + fixed", "DO-SMC (paper) + fixed")


def make_sup(cfg, kind, tag):
    if kind == "fixed":
        return FixedSupervisor(cfg)
    if kind == "rule":
        return RuleSupervisor(cfg)
    if kind in ("mpcp", "mpcv"):
        fc = "persistence" if kind == "mpcp" else "preview"
        return MPCEMSSupervisor(cfg, fc, cfg["agent"]["mpc_ems"]["horizon_s"][fc])
    return AgentSupervisor(cfg, load_actor(tag, cfg), filtered=(kind != "nofilter"),
                           version=2 if kind == "filterv2" else 1)


def job(args):
    name, ctrl, kind, tag, item, scale, seed = args[:7]
    t_start, dur = (args[7], args[8]) if len(args) > 7 else (0.0, 0.0)
    label = item if len(args) <= 7 else f"xtra:{item}"
    cfg = yaml.safe_load(open(CFG))
    if item.startswith("paper_"):
        prof = scenario_profile(transient() if item == "paper_transient" else urban())
        n_sub = 10
    else:
        prof = cycle_profile(item, cfg, ROOT)
        n_sub = 5
    sup = make_sup(cfg, kind, tag)
    le = LOG_EVERY if seed == LOG_SEED else 0
    if item == "paper_transient" and seed == LOG_SEED:
        le = 20                     # 1 ms for the short transient
    t0 = time.time()
    ep, s = run_episode(cfg, ctrl, prof, sup, seed=seed, log_every=le, overrides={"load_scale": scale},
                        n_sub=n_sub, t_start=t_start, duration=(dur if dur > 0 else None))
    s["wall_s"] = time.time() - t0
    if le:
        os.makedirs(RAW, exist_ok=True)
        stem = f"{label.replace(':', '_')}_x{scale}_{name}".replace(" ", "_").replace("+", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        d = ep.log_dict()
        np.savez_compressed(os.path.join(RAW, stem + ".npz"), **{k: v.astype(np.float32) for k, v in d.items()},
                            decisions=np.array(ep.decision_log, dtype=np.float32))
        s["raw"] = stem
    if kind in ("mpcp", "mpcv"):
        s["solve_median_s"] = float(np.median(sup.solve_times))
        s["filter_active_counts"] = sup.filter.n_active.tolist()
        s["filter_calls"] = sup.filter.n
    if kind in ("filter", "nofilter", "filterv2"):
        s["filter_active_counts"] = sup.filter.n_active.tolist()
        s["filter_calls"] = sup.filter.n
    return name, label, scale, seed, s


def _key(j):
    lab = j[4] if len(j) <= 7 else f"xtra:{j[4]}"
    return f"{j[0]}|{lab}|{j[5]}|{j[6]}"


def main():
    cfg = yaml.safe_load(open(CFG))
    items = [(c, float(sc)) for c, sc in cfg["agent"]["test_set"]]
    items_paper = [("paper_transient", 1.0), ("paper_urban", 1.0)]
    jobs = []
    for name, ctrl, kind, tag in configs(cfg):
        for it, sc in items + items_paper:
            for sd in NOISE_SEEDS:
                jobs.append((name, ctrl, kind, tag, it, sc, sd))
    # extra held-out cycles (introduced together with filter v2; never used for any design choice)
    for name, ctrl, kind, tag in configs(cfg):
        if not any(name.startswith(p) for p in EXTRA_CONFIGS):
            continue
        for it, sc, ts, du in cfg["agent"]["extra_test_set"]:
            jobs.append((name, ctrl, kind, tag, it, float(sc), LOG_SEED, float(ts), float(du)))
    jobs.sort(key=lambda j: 0 if j[4].startswith("paper") else 1)
    os.makedirs(PROC, exist_ok=True)
    part = os.path.join(PROC, "eval_partial.jsonl")   # one line per finished run (resumable)
    done = set()
    if os.path.exists(part):
        with open(part) as f:
            for line in f:
                r = json.loads(line)
                done.add(f"{r['name']}|{r['item']}|{r['scale']}|{r['seed']}")
    todo = [j for j in jobs if _key(j) not in done]
    print(f"[eval] {len(jobs)} runs, {len(done)} already done, {len(todo)} to run", flush=True)
    t0 = time.time()
    with Pool(2) as pool:
        for name, it, sc, sd, s in pool.imap_unordered(job, todo, chunksize=1):
            with open(part, "a") as f:
                f.write(json.dumps({"name": name, "item": it, "scale": sc, "seed": sd, "summary": s},
                                   default=float) + "\n")
    res = {}
    with open(part) as f:
        for line in f:
            r = json.loads(line)
            res.setdefault(r["name"], {}).setdefault(f"{r['item']}x{r['scale']}", {})[str(r["seed"])] = r["summary"]
    meta = {"noise_seeds": NOISE_SEEDS, "test_items": items, "paper_items": items_paper,
            "n_runs": len(jobs), "wall_s_this_invocation": time.time() - t0}
    with open(os.path.join(PROC, "eval_results.json"), "w") as f:
        json.dump({"meta": meta, "results": res}, f, indent=1, default=float)
    print(f"[eval] done ({time.time() - t0:.0f} s this invocation)")


if __name__ == "__main__":
    main()
