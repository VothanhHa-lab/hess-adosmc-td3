"""Parameter vector layout shared by Python and the numba kernel.

The kernel receives a flat float64 array ``p``; index constants below are
module-level ints so numba treats them as compile-time constants.
"""
import numpy as np
import yaml

NAMES = [
    # true plant
    "Cdc", "Lb", "Lsc", "rb", "rsc", "R0b", "ocv_nom", "ocv_slope", "soc_ref", "soc0",
    "Qb", "eta_b", "Csc", "Resr", "vsc0",
    # limits
    "Ibmax", "Iscmax", "vscmin", "vscmax", "dmax",
    # control
    "vref", "lam", "ks", "phi", "cv", "taub", "wci", "ki_ratio", "alloc_mode",
    "kp_pi", "ki_pi",                 # PI outer gains on (e, xi) in units of de/dt (1/s, 1/s^2)
    "Cdc_nom", "Lb_nom", "Lsc_nom",   # controller-side nominal values
    # timing / switches
    "Ts", "n_sub", "switched", "delay", "ctrl", "load_scale",
    # noise
    "sig_v", "sig_i",
    # --- extensions for the two-tier (agent-supervised) study ---
    "vsc_ref", "kE", "Prec_max",      # SC energy-recovery term in the allocation
    "k_min", "beta", "tau_f", "tau_env",  # adaptive switching gain of A-DO-SMC
]
IDX = {n: i for i, n in enumerate(NAMES)}
NP = len(NAMES)

# explicit constants for the kernel (keep in sync with NAMES — checked in tests)
(CDC, LB, LSC, RB, RSC, R0B, OCV_NOM, OCV_SLOPE, SOC_REF, SOC0,
 QB, ETA_B, CSC, RESR, VSC0,
 IBMAX, ISCMAX, VSCMIN, VSCMAX, DMAX,
 VREF, LAM, KS, PHI, CV, TAUB, WCI, KI_RATIO, ALLOC_MODE,
 KP_PI, KI_PI, CDC_NOM, LB_NOM, LSC_NOM,
 TS, N_SUB, SWITCHED, DELAY, CTRL, LOAD_SCALE,
 SIG_V, SIG_I,
 VSC_REF, KE, PREC_MAX, K_MIN, BETA, TAU_F, TAU_ENV) = range(NP)

CTRL_PI, CTRL_DOSMC, CTRL_MPC, CTRL_ADOSMC = 0, 1, 2, 3
CTRL_NAMES = {CTRL_PI: "PI", CTRL_DOSMC: "DO-SMC", CTRL_MPC: "Robust MPC", CTRL_ADOSMC: "A-DO-SMC"}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _v(cfg, sec, key):
    return float(cfg[sec][key]["value"])


def build_params(cfg, ctrl, switched=False, overrides=None):
    """Return flat parameter vector. ``overrides`` maps NAMES -> value."""
    p = np.zeros(NP)
    for k in ["Cdc", "Lb", "Lsc", "rb", "rsc", "R0b", "ocv_nom", "ocv_slope", "soc_ref",
              "soc0", "Qb", "eta_b", "Csc", "Resr", "vsc0"]:
        p[IDX[k]] = _v(cfg, "plant", k)
    for k in ["Ibmax", "Iscmax", "vscmin", "vscmax", "dmax"]:
        p[IDX[k]] = _v(cfg, "limits", k)
    for k in ["vref", "lam", "ks", "phi", "cv", "taub", "wci", "ki_ratio", "alloc_mode"]:
        p[IDX[k]] = _v(cfg, "control", k)
    # controller-side nominal plant values = nominal plant
    p[IDX["Cdc_nom"]] = _v(cfg, "plant", "Cdc")
    p[IDX["Lb_nom"]] = _v(cfg, "plant", "Lb")
    p[IDX["Lsc_nom"]] = _v(cfg, "plant", "Lsc")
    # PI gains DESIGNED for identical small-signal poles {-cv, -ks/phi} (see design.py)
    cv = p[IDX["cv"]]
    kprime = p[IDX["ks"]] / p[IDX["phi"]]
    p[IDX["kp_pi"]] = cv + kprime
    p[IDX["ki_pi"]] = cv * kprime
    p[IDX["Ts"]] = _v(cfg, "timing", "Ts")
    p[IDX["n_sub"]] = _v(cfg, "timing", "n_sub_sw" if switched else "n_sub_avg")
    p[IDX["switched"]] = 1.0 if switched else 0.0
    p[IDX["delay"]] = _v(cfg, "timing", "delay")
    p[IDX["ctrl"]] = float(ctrl)
    p[IDX["load_scale"]] = 1.0
    p[IDX["sig_v"]] = _v(cfg, "noise", "sig_v")
    p[IDX["sig_i"]] = _v(cfg, "noise", "sig_i")
    if "supervision" in cfg:
        for k in ["vsc_ref", "kE", "Prec_max", "k_min", "beta", "tau_f", "tau_env"]:
            p[IDX[k]] = _v(cfg, "supervision", k)
    if overrides:
        for k, val in overrides.items():
            p[IDX[k]] = float(val)
        if any(k in overrides for k in ("cv", "ks", "phi")) and "kp_pi" not in overrides:
            cv = p[IDX["cv"]]
            kprime = p[IDX["ks"]] / p[IDX["phi"]]
            p[IDX["kp_pi"]] = cv + kprime
            p[IDX["ki_pi"]] = cv * kprime
    return p
