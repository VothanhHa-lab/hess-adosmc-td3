#!/usr/bin/env python3
"""Train the TD3 supervisory agent (one process per training seed).

Training data  : random 30 s windows of the TRAIN cycles, randomised initial SC voltage,
                 battery SOC and load scale (domain randomisation).
Checkpoint     : selected ONLY on the VALIDATION cycles (never on the test cycles).
Variants       : --filter on  (proposed)  /  --filter off (ablation; same reward, wider action box,
                 no rate limit, no energy-reserve bound).

    python scripts/train.py --filter on  --seeds 11 22 33
    python scripts/train.py --filter off --seeds 11 22 33
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hess.cycles import cycle_profile  # noqa: E402
from hess.env import ACT_DIM, OBS_DIM, Episode, run_episode  # noqa: E402
from hess.params import CTRL_ADOSMC  # noqa: E402
from hess.supervisors import AgentSupervisor, SafetyFilter  # noqa: E402
from hess.td3 import TD3  # noqa: E402

CFG = os.path.join(ROOT, "configs", "nominal.yaml")
OUT = os.path.join(ROOT, "data", "agents")


def validate(cfg, actor, filtered, profiles, version=1):
    sup = AgentSupervisor(cfg, actor, filtered, version=version)
    rets, details = [], []
    for (name, scale), prof in profiles:
        _, s = run_episode(cfg, CTRL_ADOSMC, prof, sup, seed=777, overrides={"load_scale": scale})
        rets.append(s["return"] / max(s["n_decisions"], 1))
        details.append({"cycle": name, "scale": scale, **{k: s[k] for k in
                        ("return", "ib_rms_A", "rms_e_V", "sat_frac", "vsc_min_V", "status")}})
    return float(np.mean(rets)), details


def train_one(args):
    seed, filtered, ep_override, ev_override, suffix = args
    version = 2 if filtered == "v2" else 1
    filtered = bool(filtered)
    cfg = yaml.safe_load(open(CFG))
    if ep_override:
        cfg["agent"]["td3"]["episodes"] = ep_override
    if ev_override:
        cfg["agent"]["td3"]["eval_every"] = ev_override
    A = cfg["agent"]; hp = A["td3"]; rz = A["train_randomization"]
    rng = np.random.default_rng(seed)
    train_prof = {n: cycle_profile(n, cfg, ROOT) for n in cfg["cycles"]["train"]}
    val_prof = [((n, sc), cycle_profile(n, cfg, ROOT)) for n, sc in A["validation_set"]]
    agent = TD3(OBS_DIM, ACT_DIM, hp, seed)
    filt = SafetyFilter(cfg, filtered, version)
    kind = ("filterv2" if version == 2 else "filter") if filtered else "nofilter"
    tag = f"td3_{kind}_seed{seed}{suffix}"
    curve, vals = [], []
    best = (-np.inf, None, -1)
    total_steps = 0
    t0 = time.time()
    for epi in range(int(hp["episodes"])):
        cname = rng.choice(list(train_prof))
        prof = train_prof[cname]
        L = float(hp["episode_s"])
        ts = float(rng.uniform(0, max(prof["Tend"] - L - 1, 0)))
        ep = Episode(cfg, CTRL_ADOSMC, prof, seed=int(rng.integers(1 << 30)), t_start=ts, duration=L,
                     vsc0=float(rng.uniform(*rz["vsc0"])), soc0=float(rng.uniform(*rz["soc0"])),
                     overrides={"load_scale": float(rng.uniform(*rz["load_scale"]))})
        obs = ep.obs()
        prev = ep.theta.copy()
        R = 0.0
        done = False
        sat = 0.0
        while not done:
            if total_steps < hp["warmup"]:
                a = rng.uniform(-1, 1, ACT_DIM)
            else:
                a = agent.act(obs, noise=hp["expl_noise"])
            th = filt(prev, a, ep.info()["vb"])
            prev = th
            obs2, r, done, terms = ep.step(th)
            agent.buf.add(obs, a, r, obs2, float(ep.status != 0))
            obs = obs2
            R += r
            sat += terms[3]
            total_steps += 1
            if total_steps >= hp["warmup"]:
                agent.update()
        curve.append({"episode": epi, "cycle": str(cname), "return_per_step": R / max(ep.dec, 1),
                      "status": int(ep.status), "steps": total_steps, "wall_s": time.time() - t0})
        if (epi + 1) % int(hp["eval_every"]) == 0:
            v, det = validate(cfg, agent.actor, filtered, val_prof, version)
            vals.append({"episode": epi + 1, "val_return_per_step": v, "details": det})
            if v > best[0]:
                best = (v, agent.actor.state(), epi + 1)
            print(f"[{tag}] ep {epi + 1:4d} train R/step {np.mean([c['return_per_step'] for c in curve[-20:]]):8.3f}"
                  f"  val R/step {v:8.3f}  ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, tag + "_best.npz"), *best[1])
    np.savez(os.path.join(OUT, tag + "_last.npz"), *agent.actor.state())
    meta = {"tag": tag, "seed": seed, "filtered": filtered, "filter_version": version, "best_val_return_per_step": best[0],
            "best_episode": best[2], "episodes": int(hp["episodes"]), "total_steps": total_steps,
            "wall_time_s": time.time() - t0, "hyperparameters": hp, "reward": A["reward"],
            "train_cycles": cfg["cycles"]["train"], "validation_set": A["validation_set"],
            "filter_active_counts(tau,vsc,lam)": filt.n_active.tolist(), "filter_calls": filt.n,
            "curve": curve, "validation": vals}
    with open(os.path.join(OUT, tag + ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    return tag, best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", choices=["on", "off", "both", "v2"], default="on")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=0, help="override (pilot runs only)")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    seeds = a.seeds or cfg["agent"]["train_seeds"]
    modes = {"on": [True], "off": [False], "both": [True, False], "v2": ["v2"]}[a.filter]
    jobs = [(s, m, a.episodes, a.eval_every, a.suffix) for m in modes for s in seeds]
    with Pool(min(a.nproc, len(jobs))) as pool:
        for tag, v, e in pool.imap_unordered(train_one, jobs, chunksize=1):
            print(f"DONE {tag}: best val R/step {v:.4f} at episode {e}", flush=True)


if __name__ == "__main__":
    main()
