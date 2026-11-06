"""IEEE-style figures generated only from saved run data."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"PI": "#0072B2", "Robust MPC": "#009E73", "DO-SMC": "#D55E00"}
LS = {"PI": "-", "Robust MPC": "--", "DO-SMC": "-"}
ORDER = ["PI", "Robust MPC", "DO-SMC"]
W1 = 3.5   # IEEE single column [in]
W2 = 7.16  # double column


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 100,
    })


def save(fig, path):
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)


def fig_load(runs, path, title_t="Time (s)"):
    d = runs[ORDER[0]]
    fig, ax = plt.subplots(figsize=(W1, 1.7))
    ax.plot(d["t"], d["P_L"] / 1e3, color="k")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel(title_t); ax.set_ylabel(r"Load power $P_L$ (kW)")
    save(fig, path)


def fig_vdc(runs, path, zooms=((0.95, 1.6), (3.95, 4.6))):
    fig = plt.figure(figsize=(W2, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1, 1], wspace=0.35)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for name in ORDER:
        d = runs[name]
        for k, ax in enumerate(axs):
            ax.plot(d["t"], d["vdc"], LS[name], color=COL[name], label=name)
    for ax in axs:
        ax.axhline(400, color="k", lw=0.5, ls=":")
        ax.set_xlabel("Time (s)")
    axs[0].set_ylabel(r"$v_{dc}$ (V)")
    axs[0].legend(loc="lower left", ncol=3, frameon=True)
    for ax, (a, b) in zip(axs[1:], zooms):
        ax.set_xlim(a, b)
        ax.set_title(f"zoom {a:g}–{b:g} s")
    save(fig, path)


def fig_current(runs, path, key, ylabel, limit):
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for name in ORDER:
        d = runs[name]
        ax.plot(d["t"], d[key], LS[name], color=COL[name], label=name)
    ax.axhline(limit, color="k", ls=":", lw=0.8, label="limit")
    ax.axhline(-limit, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
    ax.legend(ncol=2, loc="best")
    save(fig, path)


def fig_disturbance(runs, path):
    fig, axs = plt.subplots(2, 1, figsize=(W1, 2.8), sharex=True)
    d = runs["DO-SMC"]
    t = d["t"]
    axs[0].plot(t, d["Dtrue"] / 1e3, "k", lw=1.4, label=r"aggregated $D$ (from plant)")
    axs[0].plot(t, d["Dhat"] / 1e3, "--", color=COL["DO-SMC"], label=r"NDO estimate $\hat D$")
    axs[0].set_ylabel(r"$D$ (kV/s)"); axs[0].legend(loc="best")
    axs[1].plot(t, (d["Dtrue"] - d["Dhat"]) / 1e3, color=COL["DO-SMC"])
    axs[1].set_ylabel(r"$\tilde D$ (kV/s)"); axs[1].set_xlabel("Time (s)")
    save(fig, path)


def fig_soc(runs, path):
    fig, axs = plt.subplots(2, 1, figsize=(W1, 2.8), sharex=True)
    for name in ORDER:
        d = runs[name]
        axs[0].plot(d["t"], 100 * d["soc_b"], LS[name], color=COL[name], label=name)
        vC = d["vC_sc"]
        axs[1].plot(d["t"], 100 * (vC - 150.0) / (300.0 - 150.0), LS[name], color=COL[name])
    axs[0].set_ylabel(r"SOC$_b$ (%)"); axs[0].legend(loc="best")
    axs[1].set_ylabel(r"SOC$_{sc}$ (%) Eq.(8)"); axs[1].set_xlabel("Time (s)")
    save(fig, path)


def fig_box(table, metric, ylabel, path):
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    data = [table[n][metric] for n in ORDER]
    bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.5)
    for patch, n in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COL[n]); patch.set_alpha(0.4)
    ax.set_ylabel(ylabel)
    save(fig, path)


def fig_sweep(xs, ys_dict, xlabel, ylabel, path, logx=True, hline=None):
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for lab, ys in ys_dict.items():
        ax.plot(xs, ys, "o-", label=lab)
    if logx:
        ax.set_xscale("log")
    if hline is not None:
        ax.axhline(hline, color="k", ls=":", lw=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(loc="best")
    save(fig, path)
