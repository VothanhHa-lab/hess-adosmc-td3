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
    # tier-1 table on the Tesla Model 3 plant: scripts/make_tier1_m3.py
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
    # F2 (tier 1 on US06, Tesla Model 3 plant): scripts/make_tier1_m3.py
    # F3: training curves (validation)
    fig, ax = plt.subplots(figsize=(plots.W1, 2.0))
    for kind, col, lab in (("filterv2", "#0072B2", "TD3 + filter v2"), ("filter", "#CC79A7", "TD3 + filter v1"),
                           ("nofilter", "#999999", "TD3, no filter")):
        for s in cfg["agent"]["train_seeds"]:
            m = json.load(open(os.path.join(AG, f"td3_{kind}_seed{s}.json")))
            ep = [v["episode"] for v in m["validation"]]; vr = [v["val_return_per_step"] for v in m["validation"]]
            ax.plot(ep, vr, "o-", ms=2.5, color=col, alpha=0.8,
                    label=lab if s == cfg["agent"]["train_seeds"][0] else None)
            if kind == "filterv2":
                kb = int(np.argmax(vr))
                ax.plot(ep[kb], vr[kb], "*", ms=7, color="#0072B2", mec="k", mew=0.4,
                        label="selected checkpoint" if s == cfg["agent"]["train_seeds"][0] else None)
    vb = json.load(open(os.path.join(PROC, "validation_baselines.json")))
    ms = json.load(open(os.path.join(PROC, "mpc_ems_selection.json")))
    hp = ms["selected_horizon_s"]["persistence"]
    for val, lab, ls in ((vb["fixed"], "fixed", ":"), (vb["rule"], "SOC rule", "--"),
                         (ms["persistence"][hp]["mean_return_per_step"], "MPC-EMS (persistence)", "-.")):
        ax.axhline(val, color="k", lw=0.8, ls=ls, label=lab)
    ax.set_xlabel("Training episode"); ax.set_ylabel("Validation return / step"); ax.legend(fontsize=5.5, ncol=2)
    lo = min(vb["fixed"], min(v["val_return_per_step"] for k in ("filterv2",) for s in cfg["agent"]["train_seeds"]
                              for v in json.load(open(os.path.join(AG, f"td3_{k}_seed{s}.json")))["validation"]))
    ax.set_ylim(1.15 * lo, 0.0)   # unfiltered points below the axis are reported in the text
    off = [v["val_return_per_step"] for s in cfg["agent"]["train_seeds"]
           for v in json.load(open(os.path.join(AG, f"td3_nofilter_seed{s}.json")))["validation"]
           if v["val_return_per_step"] < 1.15 * lo]
    if off:
        ax.annotate("no filter: %d points off scale\n(%.1f to %.1f)" % (len(off), max(off), min(off)),
                    xy=(0.5, 0.02), xycoords="axes fraction", fontsize=6, color="0.35", ha="center",
                    xytext=(0.5, 0.14), arrowprops=dict(arrowstyle="->", color="0.35", lw=0.6))
    plots.save(fig, os.path.join(FIG, "F3_validation_curves"))

    # F4: US06 time series — fixed vs rule vs proposed (median-validation seed)
    vals = {s: json.load(open(os.path.join(AG, f"td3_filterv2_seed{s}.json")))["best_val_return_per_step"]
            for s in cfg["agent"]["train_seeds"]}
    med_seed = sorted(vals, key=vals.get)[len(vals) // 2]
    vals1 = {s: json.load(open(os.path.join(AG, f"td3_filter_seed{s}.json")))["best_val_return_per_step"]
             for s in cfg["agent"]["train_seeds"]}
    med1 = sorted(vals1, key=vals1.get)[len(vals1) // 2]
    show = {"A-DO-SMC + fixed": ("fixed", "0.4", "-"), "A-DO-SMC + rule": ("rule (SOC-adaptive)", "#E69F00", "-"),
            "A-DO-SMC + MPC-EMS (persistence)": ("MPC-EMS (persistence)", "#009E73", "--"),
            f"A-DO-SMC + TD3 + filter v2 [s{med_seed}]": (f"TD3 + filter v2 (seed {med_seed})", "#0072B2", "-")}
    it = "us06x1.0"
    fig, axs = plt.subplots(5, 1, figsize=(plots.W2, 7.0), sharex=True)
    for nm, (lab, col, ls) in show.items():
        d = np.load(os.path.join(RAW, res[nm][it]["101"]["raw"] + ".npz"))
        if nm == "A-DO-SMC + fixed":
            axs[0].plot(d["t"], d["P_L"] / 1e3, color="k", lw=1.2)
        axs[1].plot(d["t"], d["vC_sc"], ls, color=col, label=lab)
        axs[1].plot(d["t"], d["vsc_ref"], ":", color=col, lw=1.2)
        axs[2].semilogy(d["t"], d["taub"], ls, color=col)
        axs[3].plot(d["t"], d["lam"], ls, color=col)
        rms = float(np.sqrt(np.mean(d["iLb"] ** 2)))
        axs[4].plot(d["t"], d["iLb"], ls, color=col, lw=1.2, label=f"{lab.split(' (')[0]}: RMS {rms:.1f} A")
        if nm == "A-DO-SMC + fixed":
            kp = int(np.argmax(d["P_L"]))
            axs[0].annotate(f"peak {d['P_L'][kp] / 1e3:.0f} kW", xy=(d["t"][kp], d["P_L"][kp] / 1e3),
                            xytext=(d["t"][kp] + 40, d["P_L"][kp] / 1e3 - 25), fontsize=6.5,
                            arrowprops=dict(arrowstyle="->", lw=0.8))
    axs[0].set_ylabel(r"$P_L$ (kW)"); axs[1].set_ylabel(r"$v_{sc}$, $v_{sc}^{ref}$ (V)")
    axs[2].set_ylabel(r"$\tau_b$ (s)"); axs[3].set_ylabel(r"$\lambda$ (1/s)")
    axs[4].set_ylabel(r"$i_b$ (A)"); axs[4].set_xlabel("Time (s)")
    hh, ll = axs[1].get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(top=0.95)
    BB = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    from hess.supervisors import SafetyFilter
    sf = SafetyFilter(cfg, True, 2)
    sf(np.array([0.2, cfg["supervision"]["vsc_ref"]["value"], 250.0]), np.zeros(3), cfg["plant"]["ocv_nom"]["value"])
    A = cfg["agent"]
    axs[1].axhline(sf.v_lo_last, color="k", lw=0.6, ls="-.")
    pen = A["reward"].get("sc_pen_lo", 190.0)
    axs[1].axhline(pen, color="r", lw=0.5, ls=":")
    axs[1].text(0.995, 0.97, "dotted lines: $v_{sc}^{ref}$ chosen by each supervisor", transform=axs[1].transAxes,
                fontsize=6, ha="right", va="top", bbox=BB)
    axs[1].text(155, sf.v_lo_last + 1.5, f"filter lower bound on $v_{{sc}}^{{ref}}$: {sf.v_lo_last:.0f} V", fontsize=5.8,
                va="bottom", bbox=BB)
    axs[1].text(400, pen - 1.5, f"lower edge of SC penalty band: {pen:.1f} V", fontsize=5.8, va="top", color="r", bbox=BB)
    # where the learned and the predictive supervisors draw the SC deepest
    axs[1].set_ylim(193, 372)
    for nm, col, tx, ty in ((f"A-DO-SMC + TD3 + filter v2 [s{med_seed}]", "#0072B2", 420, 205),
                            ("A-DO-SMC + MPC-EMS (persistence)", "#009E73", 150, 207)):
        d = np.load(os.path.join(RAW, res[nm][it]["101"]["raw"] + ".npz")); k = int(np.argmin(d["vC_sc"]))
        lab = "TD3" if "TD3" in nm else "MPC-EMS"
        plots.note(axs[1], f"{lab}: SC drawn to {d['vC_sc'][k]:.0f} V", (d["t"][k], d["vC_sc"][k]),
                   (tx, ty), color=col)

    for b in A["tau_range"]:
        axs[2].axhline(b, color="k", lw=0.5, ls="-.")
    axs[2].set_ylim(0.022, 25)
    axs[2].text(0.995, 0.975, "dash-dot: filter bounds on $\\tau_b$;  fixed: $\\tau_b=0.2$ s;  rule: slow split set by the SC SOC",
                transform=axs[2].transAxes, fontsize=5.8, ha="right", va="top", bbox=BB)
    axs[2].text(0.995, 0.025, "TD3 and MPC-EMS jump between the bounds", transform=axs[2].transAxes, fontsize=5.8,
                ha="right", va="bottom", bbox=BB)
    axs[3].axhline(cfg["control"]["lam"]["value"], color="k", lw=0.5, ls="-.")
    axs[3].text(0.995, 0.06, r"dash-dot: nominal $\lambda$ (fixed, rule, MPC-EMS)", transform=axs[3].transAxes, fontsize=6, ha="right", bbox=BB)
    dv = np.load(os.path.join(RAW, res[f"A-DO-SMC + TD3 + filter v2 [s{med_seed}]"][it]["101"]["raw"] + ".npz"))
    lam_nom = cfg["control"]["lam"]["value"]
    axs[3].set_ylim(110, 900)
    axs[3].text(0.05, 0.255, "TD3 keeps $\\lambda$ above nominal %.1f %% of the time (mean %.0f s$^{-1}$, minimum %.0f s$^{-1}$)"
                % (100 * np.mean(dv["lam"] > lam_nom), np.mean(dv["lam"]), np.min(dv["lam"])),
                transform=axs[3].transAxes, fontsize=5.8, color="#0072B2", va="center", bbox=BB, zorder=12)
    ibmax = cfg["limits"]["Ibmax"]["value"]
    axs[4].axhline(ibmax, color="k", lw=0.5, ls="-.")
    axs[4].text(0.995, 0.03, f"dash-dot: battery current limit {ibmax:.0f} A (1C); RMS values in the legend",
                transform=axs[4].transAxes, fontsize=6, ha="right", va="bottom", bbox=BB)
    for ax, lab in zip(axs, ("(a)", "(b)", "(c)", "(d)", "(e)")):
        plots.panel_label(ax, lab)
    axs[4].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6, frameon=False)
    plots.save(fig, os.path.join(FIG, "F4_us06_supervision"))

    # F5: battery RMS and saturation per test item
    items = [f"{c}x{float(s)}" for c, s in cfg["agent"]["test_set"]]
    cfgs = ["A-DO-SMC + fixed", "A-DO-SMC + rule", "A-DO-SMC + MPC-EMS (persistence)", "A-DO-SMC + MPC-EMS (preview)",
            "A-DO-SMC + TD3 no-filter", "A-DO-SMC + TD3 + filter v1", PROPOSED]
    cols = ["0.5", "#E69F00", "#009E73", "#56B4E9", "#999999", "#CC79A7", "#0072B2"]
    fig, axs = plt.subplots(1, 2, figsize=(plots.W2, 2.9))
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
        short = {"us06x1.0": "US06", "wltc_high_3bx1.0": "WLTC\nhigh", "wltc_extrahigh_3x1.0": "WLTC\nx-high",
                 "wltc_extrahigh_3x1.2": "WLTC\nx-high\n×1.2", "us06x1.1": "US06\n×1.1"}
        ax.set_xticks(x); ax.set_xticklabels([short.get(i, i) for i in items], fontsize=7); ax.set_ylabel(lab)
    axs[0].set_ylim(bottom=0)
    fixm = np.array([cfg_seed_mean(res, "A-DO-SMC + fixed", it, "ib_rms_A") for it in items])
    v2m = np.array([group_vals(res, cfg, PROPOSED, it, "ib_rms_A") for it in items]).mean(1)
    top = axs[0].get_ylim()[1]
    gmax = [max(cfg_seed_mean(res, c, it, "ib_rms_A") if c not in GROUPS else group_vals(res, cfg, c, it, "ib_rms_A").max()
                for c in cfgs) for it in items]
    for k in range(len(items)):
        axs[0].text(x[k], gmax[k] + 0.04 * top, f"{100 * (v2m[k] - fixm[k]) / fixm[k]:+.1f} %", ha="center", va="bottom",
                    fontsize=6, color="#0072B2", fontweight="bold")
    axs[0].set_ylim(0, max(gmax) + 0.16 * top)
    axs[0].text(0.09, 0.99, "labels: TD3 + filter v2 vs fixed", transform=axs[0].transAxes, fontsize=6, va="top",
                color="#0072B2")
    nof = np.array([group_vals(res, cfg, "A-DO-SMC + TD3 no-filter", it, "sat_frac") for it in items]).mean(1)
    for k in range(len(items)):
        if nof[k] > 0:
            axs[1].text(x[k] + (4 - 3) * w, nof[k], f"{nof[k]:.2f}", ha="center", va="bottom", fontsize=5.5, color="0.3")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, [t.replace("A-DO-SMC + ", "") for t in l], loc="upper center", ncol=4, fontsize=6, frameon=False)
    fig.subplots_adjust(top=0.84)
    for ax, lab in zip(axs, ("(a)", "(b)")):
        plots.panel_label(ax, lab, x=0.012, y=0.97)
    v2b = np.array(group_vals(res, cfg, PROPOSED, "us06x1.1", "sat_frac")).mean()
    axs[1].text(0.99, 0.97, "only TD3 without filter saturates on ×1.0/×1.2 cycles\n(numbers: its mean); US06×1.1: filter v2 %.2f %%\n(one deficit run, seed 33)" % v2b,
                transform=axs[1].transAxes, fontsize=6, ha="right", va="top", bbox=plots.BOX)
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
