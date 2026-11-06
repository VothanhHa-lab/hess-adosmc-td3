#!/usr/bin/env python3
"""Checks specific to the two-tier study (run after training).
    python tests/verify_agent.py   -> results/verification_agent.json
"""
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hess.cycles import cycle_profile, scenario_profile  # noqa: E402
from hess.env import Episode, run_episode  # noqa: E402
from hess.params import CTRL_ADOSMC, CTRL_DOSMC, CTRL_MPC, CTRL_PI, load_config  # noqa: E402
from hess.scenarios import transient  # noqa: E402
from hess.sim import run as run_old  # noqa: E402
from hess.supervisors import FixedSupervisor, SafetyFilter  # noqa: E402
from hess.td3 import MLP  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "nominal.yaml")))
R = {}


def check(name, ok, **info):
    R[name] = {"pass": bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
                                     for k, v in info.items()}}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {info}")


# 1. chunked kernel reproduces the verified single-run kernel (noise-free, n_sub=10)
sc = transient()
prof = scenario_profile(sc, dt=1e-4)
for c in (CTRL_PI, CTRL_DOSMC, CTRL_MPC):
    d_old, _ = run_old(load_config(os.path.join(ROOT, "configs", "nominal.yaml")), c, sc, noise=False)
    old = float(np.max(np.abs(d_old["vdc"][d_old["t"] > 0.5] - 400)))
    _, s = run_episode(cfg, c, prof, FixedSupervisor(cfg), seed=1, n_sub=10,
                       overrides={"sig_v": 0.0, "sig_i": 0.0})
    # chunk kernel max over t>=0 (idle part has |e|~0) ; table interpolation of the load (0.1 ms)
    check(f"chunk_kernel_matches_original_ctrl{c}", abs(s["max_dev_V"] - old) / old < 0.01,
          original=old, chunked=s["max_dev_V"])

# 2. energy balance inside chunk kernel
ep = Episode(cfg, CTRL_ADOSMC, cycle_profile("us06", cfg, ROOT), seed=3, duration=60.0)
while ep.dec < ep.ndec:
    ep.step(None)
x = ep.x
C, Lb, Lsc = cfg["plant"]["Cdc"]["value"], cfg["plant"]["Lb"]["value"], cfg["plant"]["Lsc"]["value"]
dE = 0.5 * C * x[0] ** 2 + 0.5 * Lb * x[1] ** 2 + 0.5 * Lsc * x[2] ** 2 - 0.5 * C * 400 ** 2
net = x[5] + x[6] - x[7] - x[8]
rel = abs(dE - net) / (abs(x[5]) + abs(x[6]) + abs(x[8]))
check("energy_balance_chunk_kernel_us06_60s", rel < 1e-6, rel_residual=rel)

# 3. bumpless observer-gain change: D_hat continuous across lambda switch
ep = Episode(cfg, CTRL_ADOSMC, cycle_profile("us06", cfg, ROOT), seed=4, duration=5.0)
for _ in range(20):
    ep.step(None)
D_before = ep.dhat()
ep.apply_theta(np.array([0.2, 250.0, 700.0]))
D_after = ep.dhat()
check("bumpless_lambda_switch", abs(D_after - D_before) < 1e-9 * (1 + abs(D_before)),
      before=D_before, after=D_after)

# 4. safety filter properties on random actions
f = SafetyFilter(cfg, True)
rng = np.random.default_rng(0)
prev = np.array([0.2, 250.0, 250.0])
ok = True
A = cfg["agent"]
for _ in range(5000):
    th = f(prev, rng.uniform(-1, 1, 3), 296.0)
    ok &= abs(np.log(th[0]) - np.log(prev[0])) <= A["rate_log_tau"] + 1e-12
    ok &= abs(th[1] - prev[1]) <= A["rate_vsc"] + 1e-9 or th[1] in (A["vsc_range"][0], A["vsc_range"][1])
    ok &= abs(th[2] - prev[2]) <= A["rate_lam"] + 1e-9
    ok &= A["tau_range"][0] - 1e-12 <= th[0] <= A["tau_range"][1] + 1e-12
    ok &= A["vsc_range"][0] - 1e-9 <= th[1] <= A["vsc_range"][1] + 1e-9
    ok &= A["lam_range"][0] - 1e-9 <= th[2] <= A["lam_range"][1] + 1e-9
    ok &= th[2] * cfg["timing"]["Ts"]["value"] <= 0.04 + 1e-12
    prev = th
check("safety_filter_box_rate_and_discretisation_bounds", ok)

# 5. A-DO-SMC enforces Eq.(38) approximately: ks(t) >= |D~(t)| (noise-free transient)
ep, s = run_episode(cfg, CTRL_ADOSMC, prof, FixedSupervisor(cfg), seed=1, n_sub=10, log_every=1,
                    overrides={"sig_v": 0.0, "sig_i": 0.0})
d = ep.log_dict()
w = d["t"] > 0.2
Dt = np.abs(d["Dtrue"][w] - d["Dhat"][w])
ratio = d["ks"][w] / np.maximum(Dt, 1e-9)
frac = float(np.mean(d["ks"][w] >= Dt))
check("adaptive_gain_satisfies_eq38_fraction_of_time", frac > 0.95, fraction=frac,
      min_ratio=float(np.min(ratio)), max_Dtilde=float(Dt.max()), max_ks=float(d["ks"][w].max()))

# 6. determinism
_, a = run_episode(cfg, CTRL_ADOSMC, cycle_profile("udds", cfg, ROOT), FixedSupervisor(cfg), seed=9, duration=20)
_, b = run_episode(cfg, CTRL_ADOSMC, cycle_profile("udds", cfg, ROOT), FixedSupervisor(cfg), seed=9, duration=20)
check("determinism_episode", a == b)

# 7. plant step convergence for the long-cycle setting (n_sub 5 vs 20), A-DO-SMC on 60 s of US06
r = {}
for ns in (5, 20):
    _, s = run_episode(cfg, CTRL_ADOSMC, cycle_profile("us06", cfg, ROOT), FixedSupervisor(cfg), seed=2,
                       duration=60, n_sub=ns, overrides={"sig_v": 0.0, "sig_i": 0.0})
    r[ns] = s
rel = max(abs(r[5][k] - r[20][k]) / abs(r[20][k]) for k in ("max_dev_V", "ib_rms_A", "rms_e_V"))
check("timestep_convergence_nsub5_vs_20", rel < 0.02, rel_change=rel,
      maxdev=(r[5]["max_dev_V"], r[20]["max_dev_V"]))

# 8. TD3 gradients vs finite differences
m = MLP([5, 8, 8, 2], np.random.default_rng(1), "tanh")
x = rng.normal(size=(4, 5)); wv = rng.normal(size=(4, 2))
m.forward(x); g, dx = m.backward(wv)
P = m.params(); eps = 1e-6; errs = []
for pi in range(len(P)):
    idx = tuple(0 for _ in P[pi].shape)
    old = P[pi][idx]
    P[pi][idx] = old + eps; fp = (m.forward(x) * wv).sum()
    P[pi][idx] = old - eps; fm = (m.forward(x) * wv).sum()
    P[pi][idx] = old
    errs.append(abs(g[pi][idx] - (fp - fm) / (2 * eps)))
check("mlp_backprop_finite_difference", max(errs) < 1e-6, max_abs_err=max(errs))

with open(os.path.join(ROOT, "results", "verification_agent.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=str)
nf = sum(not v["pass"] for v in R.values())
print(f"\n{len(R) - nf}/{len(R)} checks passed")
