#!/usr/bin/env python3
"""Tables, statistics and figures for the two-tier study — reads saved data only.
    python scripts/report.py
"""
import json
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hess import plots  # noqa: E402
from hess.design import design_mpc, dummy_mpc  # noqa: E402
from hess.kernel_sup import bench_step2  # noqa: E402
from hess.params import CTRL_ADOSMC, CTRL_DOSMC, CTRL_MPC, CTRL_PI, build_params  # noqa: E402
from hess.supervisors import SafetyFilter  # noqa: E402
from hess.td3 import MLP  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

CFG = os.path.join(ROOT, "configs", "nominal.yaml")
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw", "eval")
AG = os.path.join(ROOT, "data", "agents")
FIG = os.path.join(ROOT, "results", "figures")
TAB = os.path.join(ROOT, "results", "tables")

BASE = ["PI + fixed", "DO-SMC (paper) + fixed", "Robust MPC + fixed", "A-DO-SMC + fixed", "A-DO-SMC + rule"]
GROUPS = {"A-DO-SMC + TD3 no-filter": "A-DO-SMC + TD3 no-filter [s{}]",
          "A-DO-SMC + TD3 + filter v1": "A-DO-SMC + TD3 + filter [s{}]",
          "A-DO-SMC + TD3 + filter v2 (proposed)": "A-DO-SMC + TD3 + filter v2 [s{}]"}
PROPOSED = "A-DO-SMC + TD3 + filter v2 (proposed)"
KEYS = [("max_dev_V", "max|e| (V)", 2), ("rms_e_V", "RMS e (V)", 3), ("ib_rms_A", "RMS i_b (A)", 2),
        ("ib_slew_rms_Aps", "RMS di_b/dt (A/s)", 1), ("ib_Ah", "Ah_b", 3), ("sat_frac", "sat. frac (%)", 3),
        ("vsc_min_V", "min v_sc (V)", 1)]


def write_table(rows, hdr, stem):
    import csv
    with open(stem + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(c) for c in r) + " |\n")


def cfg_seed_mean(res, name, item, key):
    """mean over noise seeds for one configuration name"""
    d = res[name][item]
    v = np.array([d[s][key] for s in d], float)
    if key == "sat_frac":
        v = 100 * v
    return float(np.mean(v))


def group_vals(res, cfg, gname, item, key):
    """per-training-seed values (each = mean over noise seeds)"""
    return np.array([cfg_seed_mean(res, GROUPS[gname].format(s), item, key) for s in cfg["agent"]["train_seeds"]])


def tables(cfg, E):
    res = E["results"]
    items = [f"{c}x{float(s)}" for c, s in cfg["agent"]["test_set"]]
    paper = ["paper_transientx1.0", "paper_urbanx1.0"]
    # ---- tier-1 on the GTSD profiles
    rows = []
    for name in ["PI + fixed", "DO-SMC (paper) + fixed", "Robust MPC + fixed", "A-DO-SMC + fixed"]:
        r = [name]
        for it in paper:
            r += [f"{cfg_seed_mean(res, name, it, 'max_dev_V'):.2f}", f"{cfg_seed_mean(res, name, it, 'rms_e_V'):.3f}"]
        rows.append(r)
    write_table(rows, ["Controller", "Fig.2 max|e| (V)", "Fig.2 RMS e (V)", "Fig.7 max|e| (V)", "Fig.7 RMS e (V)"],
                os.path.join(TAB, "T2_tier1_gtsd_profiles"))
    # ---- test cycles: full table
    for it in items:
        rows = []
        for name in BASE:
            rows.append([name] + [f"{cfg_seed_mean(res, name, it, k):.{nd}f}" for k, _, nd in KEYS] +
                        [_status(res, name, it)])
        for g in GROUPS:
            r = [g + " (mean ± std, 3 train seeds)"]
            for k, _, nd in KEYS:
                v = group_vals(res, cfg, g, it, k)
                r.append(f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}")
            r.append(",".join(_status(res, GROUPS[g].format(s), it) for s in cfg["agent"]["train_seeds"]))
            rows.append(r)
        write_table(rows, ["Configuration"] + [h for _, h, _ in KEYS] + ["status"],
                    os.path.join(TAB, f"T3_test_{it}"))
    # ---- paired comparison of the proposed agent vs rule and vs fixed (A-DO-SMC tier-1 for all)
    rows = []
    summary = {}
    for it in items:
        for ref in ("A-DO-SMC + fixed", "A-DO-SMC + rule"):
            for k in ("ib_rms_A", "ib_slew_rms_Aps", "sat_frac", "return"):
                deltas, wins, n = [], 0, 0
                for s in cfg["agent"]["train_seeds"]:
                    a = res[GROUPS[PROPOSED].format(s)][it]
                    b = res[ref][it]
                    for sd in a:
                        va, vb = a[sd][k], b[sd][k]
                        if k == "return":
                            d = va - vb             # higher is better
                            wins += d > 0
                            deltas.append(d)
                        elif k == "sat_frac":
                            d = 100 * (va - vb)     # percentage points
                            wins += d < 0
                            deltas.append(d)
                        else:
                            d = 100 * (va - vb) / vb  # % change, lower is better
                            wins += d < 0
                            deltas.append(d)
                        n += 1
                deltas = np.array(deltas)
                summary[(it, ref, k)] = (deltas.mean(), deltas.min(), deltas.max(), wins, n)
                unit = {"return": "", "sat_frac": " pp", "ib_rms_A": " %", "ib_slew_rms_Aps": " %"}[k]
                rows.append([it, ref, k, f"{deltas.mean():+.2f}{unit}", f"[{deltas.min():+.2f}, {deltas.max():+.2f}]",
                             f"{wins}/{n}"])
    write_table(rows, ["test item", "reference", "metric", "mean Δ (proposed − ref)", "[min, max]",
                       "wins (proposed better)"], os.path.join(TAB, "T4_paired_proposed_vs_refs"))
    # ---- filter vs no filter (dispersion across seeds + violations)
    rows = []
    for it in items:
        r = [it]
        for g in GROUPS:
            ret = group_vals(res, cfg, g, it, "return")
            sat = group_vals(res, cfg, g, it, "sat_frac")
            vmin = group_vals(res, cfg, g, it, "vsc_min_V")
            r += [f"{ret.mean():.1f} [{ret.min():.1f}, {ret.max():.1f}]", f"{sat.max():.3f}", f"{vmin.min():.1f}"]
        rows.append(r)
    hdr = ["test item"]
    for g in GROUPS:
        hdr += [f"{g}: return mean [min,max]", f"{g}: max sat (%)", f"{g}: min v_sc (V)"]
    write_table(rows, hdr, os.path.join(TAB, "T5_filter_ablation"))
    # extra held-out cycles (one noise seed)
    xitems = sorted({k for nm in res for k in res[nm] if k.startswith("xtra:")})
    for it in xitems:
        rows = []
        for name in ["DO-SMC (paper) + fixed", "Robust MPC + fixed", "A-DO-SMC + fixed", "A-DO-SMC + rule"]:
            if it in res.get(name, {}):
                rows.append([name] + [f"{cfg_seed_mean(res, name, it, k):.{nd}f}" for k, _, nd in KEYS] + [_status(res, name, it)])
        for g in ("A-DO-SMC + TD3 + filter v1", PROPOSED):
            if all(it in res.get(GROUPS[g].format(s), {}) for s in cfg["agent"]["train_seeds"]):
                r = [g + " (mean ± std, 3 train seeds)"]
                for k, _, nd in KEYS:
                    v = group_vals(res, cfg, g, it, k)
                    r.append(f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}")
                r.append(",".join(_status(res, GROUPS[g].format(s), it) for s in cfg["agent"]["train_seeds"]))
                rows.append(r)
        write_table(rows, ["Configuration"] + [h for _, h, _ in KEYS] + ["status"],
                    os.path.join(TAB, f"T8_extra_{it.replace('xtra:', '')}"))
    return items, summary


def _status(res, name, it):
    st = [res[name][it][s]["status"] for s in res[name][it]]
    return "ok" if all(x == "ok" for x in st) else f"{sum(x != 'ok' for x in st)} failed"


def timing(cfg):
    out = {}
    N = int(cfg["control"]["mpc_N"]["value"]); wmax = float(cfg["control"]["mpc_wmax"]["value"])
    for c, nm in ((CTRL_PI, "PI"), (CTRL_DOSMC, "DO-SMC (paper)"), (CTRL_MPC, "Robust MPC"), (CTRL_ADOSMC, "A-DO-SMC")):
        p = build_params(cfg, c)
        m = design_mpc(p, N, wmax) if c == CTRL_MPC else dummy_mpc()
        bench_step2(p, 1000, m["H"], m["Hinv"], m["F"], m["margin"], 0)
        ts = []
        for r in range(7):
            t0 = time.perf_counter(); bench_step2(p, 200000, m["H"], m["Hinv"], m["F"], m["margin"], r)
            ts.append((time.perf_counter() - t0) / 200000 * 1e6)
        out[nm + " (tier-1 update, us)"] = float(np.median(ts))
    H = cfg["agent"]["td3"]["hidden"]
    actor = MLP([11, H, H, 3], np.random.default_rng(0), "tanh")
    f = SafetyFilter(cfg, True)
    o = np.zeros(11); prev = np.array([0.2, 250.0, 250.0])
    ts = []
    for r in range(5):
        t0 = time.perf_counter()
        for _ in range(5000):
            a = actor.forward(o[None, :])[0]
            prev = f(prev, a, 296.0)
        ts.append((time.perf_counter() - t0) / 5000 * 1e6)
    out["TD3 actor + safety filter (tier-2 decision, us, NumPy/Python)"] = float(np.median(ts))
    out["n_actor_parameters"] = int(sum(p.size for p in actor.params()))
    out["note"] = "x86_64 server CPU; tier-1 = compiled numba loop; tier-2 = interpreted Python. Relative indicator only."
    return out


def figures(cfg, E):
    plots.style()
    res = E["results"]
    # F2: tier-1 on GTSD transient
    names = {"DO-SMC (paper) + fixed": ("DO-SMC [GTSD]", "#D55E00", "-"),
             "A-DO-SMC + fixed": ("A-DO-SMC (proposed tier 1)", "#0072B2", "-"),
             "Robust MPC + fixed": ("Robust MPC", "#009E73", (0, (3, 2)))}
    fig, axs = plt.subplots(2, 2, figsize=(plots.W2, 3.4), gridspec_kw={"width_ratios": [2, 1]})
    for nm, (lab, col, ls) in names.items():
        stem = res[nm]["paper_transientx1.0"]["101"]["raw"]
        d = np.load(os.path.join(RAW, stem + ".npz"))
        for ax in (axs[0, 0], axs[0, 1]):
            ax.plot(d["t"], d["vdc"], linestyle=ls, color=col, label=lab, lw=1.3 if nm.startswith("Robust") else 1.0)
        if nm == "A-DO-SMC + fixed":
            axs[1, 0].semilogy(d["t"], d["ks"], color=col, label=r"$k_s(t)$")
            axs[1, 0].semilogy(d["t"], np.abs(d["Dtrue"] - d["Dhat"]) + 1, color="0.5", lw=0.6, label=r"$|\tilde D(t)|$")
            axs[1, 1].plot(d["t"], d["phi"], color=col)
    axs[0, 1].set_xlim(0.95, 1.6)
    axs[0, 0].set_ylabel(r"$v_{dc}$ (V)")
    axs[0, 0].legend(loc="lower center", bbox_to_anchor=(0.75, 1.02), ncol=3, frameon=False)
    axs[1, 0].set_ylabel("V/s"); axs[1, 0].legend(loc="upper right", ncol=2)
    axs[1, 1].set_ylabel(r"$\phi_s(t)$ (V)")
    for ax in axs[1]:
        ax.set_xlabel("Time (s)")
    plots.save(fig, os.path.join(FIG, "F2_tier1_gtsd_transient"))

    # F3: training curves (validation)
    fig, ax = plt.subplots(figsize=(plots.W1, 2.0))
    for kind, col, lab in (("filterv2", "#0072B2", "TD3 + filter v2"), ("filter", "#CC79A7", "TD3 + filter v1"),
                           ("nofilter", "#999999", "TD3, no filter")):
        for s in cfg["agent"]["train_seeds"]:
            m = json.load(open(os.path.join(AG, f"td3_{kind}_seed{s}.json")))
            ep = [v["episode"] for v in m["validation"]]; vr = [v["val_return_per_step"] for v in m["validation"]]
            ax.plot(ep, vr, "o-", ms=2.5, color=col, alpha=0.8,
                    label=lab if s == cfg["agent"]["train_seeds"][0] else None)
    ax.set_xlabel("Training episode"); ax.set_ylabel("Validation return / step"); ax.legend(fontsize=6)
    ax.set_ylim(-0.65, -0.12)   # points below the axis are reported in Table T6 / text
    plots.save(fig, os.path.join(FIG, "F3_validation_curves"))

    # F4: US06 time series — fixed vs rule vs proposed (median-validation seed)
    vals = {s: json.load(open(os.path.join(AG, f"td3_filterv2_seed{s}.json")))["best_val_return_per_step"]
            for s in cfg["agent"]["train_seeds"]}
    med_seed = sorted(vals, key=vals.get)[len(vals) // 2]
    vals1 = {s: json.load(open(os.path.join(AG, f"td3_filter_seed{s}.json")))["best_val_return_per_step"]
             for s in cfg["agent"]["train_seeds"]}
    med1 = sorted(vals1, key=vals1.get)[len(vals1) // 2]
    show = {"A-DO-SMC + fixed": ("fixed", "0.4", "-"), "A-DO-SMC + rule": ("rule (SOC-adaptive)", "#E69F00", "-"),
            f"A-DO-SMC + TD3 + filter [s{med1}]": (f"TD3 + filter v1 (seed {med1})", "#CC79A7", "--"),
            f"A-DO-SMC + TD3 + filter v2 [s{med_seed}]": (f"TD3 + filter v2 (seed {med_seed})", "#0072B2", "-")}
    it = "us06x1.0"
    fig, axs = plt.subplots(5, 1, figsize=(plots.W2, 6.2), sharex=True)
    for nm, (lab, col, ls) in show.items():
        d = np.load(os.path.join(RAW, res[nm][it]["101"]["raw"] + ".npz"))
        if nm == "A-DO-SMC + fixed":
            axs[0].plot(d["t"], d["P_L"] / 1e3, color="k", lw=0.7)
        axs[1].plot(d["t"], d["vC_sc"], ls, color=col, label=lab)
        axs[1].plot(d["t"], d["vsc_ref"], ":", color=col, lw=0.8)
        axs[2].semilogy(d["t"], d["taub"], ls, color=col)
        axs[3].plot(d["t"], d["lam"], ls, color=col)
        axs[4].plot(d["t"], d["iLb"], ls, color=col, lw=0.6)
    axs[0].set_ylabel(r"$P_L$ (kW)"); axs[1].set_ylabel(r"$v_{sc}$, $v_{sc}^{ref}$ (V)")
    axs[1].legend(loc="lower left", ncol=3); axs[2].set_ylabel(r"$\tau_b$ (s)"); axs[3].set_ylabel(r"$\lambda$ (1/s)")
    axs[4].set_ylabel(r"$i_b$ (A)"); axs[4].set_xlabel("Time (s)")
    axs[1].legend(loc="lower left", ncol=4, fontsize=6)
    plots.save(fig, os.path.join(FIG, "F4_us06_supervision"))

    # F5: battery RMS and saturation per test item
    items = [f"{c}x{float(s)}" for c, s in cfg["agent"]["test_set"]]
    cfgs = ["A-DO-SMC + fixed", "A-DO-SMC + rule", "A-DO-SMC + MPC-EMS (persistence)", "A-DO-SMC + MPC-EMS (preview)",
            "A-DO-SMC + TD3 no-filter", "A-DO-SMC + TD3 + filter v1", PROPOSED]
    cols = ["0.5", "#E69F00", "#009E73", "#56B4E9", "#999999", "#CC79A7", "#0072B2"]
    fig, axs = plt.subplots(1, 2, figsize=(plots.W2, 2.5))
    x = np.arange(len(items)); w = 0.12
    for j, (c, col) in enumerate(zip(cfgs, cols)):
        for ax, key in zip(axs, ("ib_rms_A", "sat_frac")):
            if c in GROUPS:
                v = np.array([group_vals(res, cfg, c, it, key) for it in items])
                m, lo, hi = v.mean(1), v.min(1), v.max(1)
                ax.bar(x + (j - 3) * w, m, w, color=col, label=c, yerr=[m - lo, hi - m], capsize=1.5, error_kw={"lw": 0.6})
            else:
                m = np.array([cfg_seed_mean(res, c, it, key) for it in items])
                ax.bar(x + (j - 3) * w, m, w, color=col, label=c)
    for ax, lab in zip(axs, ("RMS battery current (A)", "Saturated time (%)")):
        short = {"us06x1.0": "US06", "wltc_high_3bx1.0": "WLTC-H", "wltc_extrahigh_3x1.0": "WLTC-XH",
                 "wltc_extrahigh_3x1.2": "WLTC-XH\n×1.2", "us06x1.1": "US06\n×1.1"}
        ax.set_xticks(x); ax.set_xticklabels([short.get(i, i) for i in items], fontsize=7); ax.set_ylabel(lab)
    axs[0].set_ylim(bottom=0)
    axs[1].legend(fontsize=5.5, loc="upper left")
    plots.save(fig, os.path.join(FIG, "F5_test_battery_rms_saturation"))
    return med_seed


def main():
    cfg = yaml.safe_load(open(CFG))
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    E = json.load(open(os.path.join(PROC, "eval_results.json")))
    items, summary = tables(cfg, E)
    med = figures(cfg, E)
    tim = timing(cfg)
    with open(os.path.join(PROC, "timing_two_tier.json"), "w") as f:
        json.dump(tim, f, indent=1)
    # training meta table
    rows = []
    for kind in ("filterv2", "filter", "nofilter"):
        for s in cfg["agent"]["train_seeds"]:
            m = json.load(open(os.path.join(AG, f"td3_{kind}_seed{s}.json")))
            rows.append([m["tag"], m["best_episode"], f"{m['best_val_return_per_step']:.4f}", m["total_steps"],
                         f"{m['wall_time_s'] / 60:.1f}", str(m["filter_active_counts(tau,vsc,lam)"]), m["filter_calls"]])
    write_table(rows, ["agent", "best episode (val)", "best val return/step", "env steps", "wall (min)",
                       "filter active (tau,vsc,lam)", "filter calls"], os.path.join(TAB, "T6_training"))
    write_table([[k, f"{v:.3f}" if isinstance(v, float) else v] for k, v in tim.items()], ["item", "value"],
                os.path.join(TAB, "T7_timing"))
    print("[report] median-validation seed shown in F4:", med)


if __name__ == "__main__":
    main()
