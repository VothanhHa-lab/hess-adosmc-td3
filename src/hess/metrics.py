"""Performance indices.  All computed from the TRUE plant signals in the log
(not from noisy measurements).  Definitions (units in brackets):

max_dev        max |v_dc - v_ref| in the window                                   [V]
settle_time    smallest T>=0 such that |v_dc - v_ref| <= band for all t in
               [t_ev+T, t_win_end]; band = BAND_V; 0 if never left the band;
               NaN if not settled before window end                              [s]
rms_e          sqrt(mean(e^2)) over the window                                    [V]
ib_peak        max |i_Lb| over the window                                         [A]
ib_violation   max(|i_Lb| - Ib_max, 0) and fraction of time with |i_Lb| > Ib_max*(1+TOL)
ib_slew_max    max |d i_Lb/dt| after a 1 ms moving average                        [A/s]
isc_recovery   after the event, time from the |i_Lsc| peak until |i_Lsc| stays
               below RECOV_FRAC * peak until the window end; NaN if never         [s]
ripple_pp      median per-switching-period peak-to-peak inductor current
               (switched-PWM runs only)                                          [A]
"""
import numpy as np

BAND_V = 2.0          # 0.5 % of 400 V
TOL = 0.01            # 1 % tolerance for "constraint violation"
RECOV_FRAC = 0.10


def settle_time(t, e, t0, t1, band=BAND_V):
    m = (t >= t0) & (t <= t1)
    tt, ee = t[m], np.abs(e[m])
    if tt.size == 0:
        return np.nan
    out = ee > band
    if not out.any():
        return 0.0
    last = np.nonzero(out)[0][-1]
    if last == tt.size - 1:
        return np.nan
    return float(tt[last + 1] - t0)


def recovery_time(t, i, t0, t1, frac=RECOV_FRAC):
    m = (t >= t0) & (t <= t1)
    tt, ii = t[m], np.abs(i[m])
    if tt.size == 0:
        return np.nan
    k = int(np.argmax(ii))
    pk = ii[k]
    if pk <= 1e-9:
        return 0.0
    above = np.nonzero(ii[k:] > frac * pk)[0]
    last = k + above[-1]
    if last == tt.size - 1:
        return np.nan
    return float(tt[last + 1] - tt[k])


def moving_avg(x, n):
    if n <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, y[0]), y])


def max_slew(t, i, win_s=1e-3):
    dt = float(np.median(np.diff(t)))
    n = max(1, int(round(win_s / dt)))
    y = moving_avg(i, n)
    return float(np.max(np.abs(np.diff(y))) / dt)


def window_metrics(d, t0, t1, Ibmax, Iscmax, vref=400.0):
    t = d["t"]
    e = d["vdc"] - vref
    m = (t >= t0) & (t <= t1)
    if m.sum() < 3:      # e.g. run failed before this window
        keys = ["max_dev_V", "settle_s", "rms_e_V", "ib_peak_A", "isc_peak_A", "ib_over_A",
                "isc_over_A", "ib_frac_viol", "isc_frac_viol", "isc_recovery_s", "ib_slew_max_Aps"]
        return {k: float("nan") for k in keys}
    ib = np.abs(d["iLb"][m]); isc = np.abs(d["iLsc"][m])
    return {
        "max_dev_V": float(np.max(np.abs(e[m]))),
        "settle_s": settle_time(t, e, t0, t1),
        "rms_e_V": float(np.sqrt(np.mean(e[m] ** 2))),
        "ib_peak_A": float(ib.max()),
        "isc_peak_A": float(isc.max()),
        "ib_over_A": float(max(ib.max() - Ibmax, 0.0)),
        "isc_over_A": float(max(isc.max() - Iscmax, 0.0)),
        "ib_frac_viol": float(np.mean(ib > Ibmax * (1 + TOL))),
        "isc_frac_viol": float(np.mean(isc > Iscmax * (1 + TOL))),
        "isc_recovery_s": recovery_time(t, d["iLsc"], t0, t1),
        "ib_slew_max_Aps": max_slew(t[m], d["iLb"][m]),
    }


def run_metrics(d, meta, scen):
    """Global + per-event metrics for one run."""
    p = meta["params"]
    Ib, Isc, vref = p["Ibmax"], p["Iscmax"], p["vref"]
    t = d["t"]
    tend = float(t[-1])
    out = {"status": meta["status"]}
    out["global"] = window_metrics(d, 0.5, tend, Ib, Isc, vref)
    out["global"]["constraint_violation"] = bool(out["global"]["ib_frac_viol"] > 0 or
                                                 out["global"]["isc_frac_viol"] > 0)
    out["t_end_reached_s"] = tend
    dt = float(np.median(np.diff(t)))
    out["global"]["ib_rms_A"] = float(np.sqrt(np.mean(d["iLb"] ** 2)))
    out["global"]["ib_Ah_throughput"] = float(np.sum(np.abs(d["iLb"])) * dt / 3600.0)
    out["global"]["soc_b_final"] = float(d["soc_b"][-1])
    out["global"]["vsc_min_V"] = float(np.min(d["vC_sc"]))
    out["global"]["vsc_final_V"] = float(d["vC_sc"][-1])
    out["global"]["u_sat_frac"] = float(np.mean(np.abs(d["u"] - d["u_un"]) > 1e-9))
    out["global"]["qp_active_frac"] = float(np.mean(d["qp_iter"] > 0))
    if "Dhat" in d and meta["ctrl_id"] != 0:
        w = t >= 0.5
        out["global"]["Dtilde_max_Vps"] = (float(np.max(np.abs(d["Dtrue"][w] - d["Dhat"][w])))
                                           if w.any() else float("nan"))
    evs = sorted(scen["events"].items(), key=lambda kv: kv[1])
    out["events"] = {}
    for i, (name, t0) in enumerate(evs):
        # window ends 0.1 s before the next event: the smooth (tanh) load transition of the
        # next event starts slightly before its nominal time and must not be attributed here
        t1 = evs[i + 1][1] - 0.1 if i + 1 < len(evs) else tend
        out["events"][name] = window_metrics(d, t0, t1, Ib, Isc, vref)
    if meta["switched"]:
        r = {}
        for nm, (a, b) in {"idle_0.3-0.9s": (0.3, 0.9), "drive_2.0-3.5s": (2.0, 3.5),
                           "regen_4.7-5.3s": (4.7, 5.3)}.items():
            mm = (t >= a) & (t <= b)
            if mm.any():
                r[nm] = {"rip_b_pp_A": float(np.median(d["rip_b_pp"][mm])),
                         "rip_sc_pp_A": float(np.median(d["rip_sc_pp"][mm]))}
        out["ripple"] = r
    return out


def energy_balance(meta, Cdc, Lb, Lsc, vref, vsc0):
    """Relative residual of  dE_stored = E_b + E_sc - E_r - E_load  (averaged or switched)."""
    x = np.array(meta["final_state"])
    vdc, iLb, iLsc = x[0], x[1], x[2]
    E0 = 0.5 * Cdc * vref ** 2
    E1 = 0.5 * Cdc * vdc ** 2 + 0.5 * Lb * iLb ** 2 + 0.5 * Lsc * iLsc ** 2
    rhs = x[5] + x[6] - x[7] - x[8]
    scale = abs(x[5]) + abs(x[6]) + abs(x[8]) + 1e-9
    return {"dE_stored_J": E1 - E0, "net_in_J": rhs, "abs_residual_J": abs((E1 - E0) - rhs),
            "rel_residual": abs((E1 - E0) - rhs) / scale}
