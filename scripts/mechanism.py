#!/usr/bin/env python3
"""Diagnose how the supervisors use the SC energy-recovery term (from stored decimated eval logs, seed 101)."""
import glob, json, os
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "nominal.yaml")))
kE = cfg["supervision"]["kE"]["value"]; Csc = cfg["plant"]["Csc"]["value"]; Pm = cfg["supervision"]["Prec_max"]["value"]
out = {}
for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "eval", "*.npz"))):
    b = os.path.basename(f)[:-4]
    if not any(k in b for k in ("A-DO-SMC__fixed", "filter_v2_s", "A-DO-SMC__rule", "MPC-EMS")):
        continue
    if not b.startswith(("us06_x1.0", "wltc_high", "xtra_long", "wltc_extrahigh_3_x1.0", "xtra_ftp", "xtra_wmtc", "wltc_extrahigh_3_x1.2")):
        continue
    d = np.load(f); vr = d["vsc_ref"]; vc = d["vC_sc"]
    P = np.clip(kE * 0.5 * Csc * (vr ** 2 - vc ** 2), -Pm, Pm)   # vC used in place of measured terminal voltage
    out[b] = {"Prec_mean_W": float(P.mean()), "Prec_absmean_W": float(np.abs(P).mean()),
              "frac_cap": float(np.mean(np.abs(P) > 0.9998 * Pm)), "vsc_ref_std": float(vr.std()),
              "tau_median": float(np.median(d["taub"])), "vsc_ref_mean": float(vr.mean()),
              "frac_tau_min": float(np.mean(d["taub"] < 0.0501))}
json.dump(out, open(os.path.join(ROOT, "data", "processed", "mechanism_diagnostics.json"), "w"), indent=1)
print(len(out), "runs diagnosed")
