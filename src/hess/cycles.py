"""Standard drive cycles -> traction power tables.

Speed profiles (1 Hz) come from the NREL FASTSim repository (pinned commit in
data/cycles/SOURCE_COMMIT.txt).  Traction power at the DC bus:

    F = m_eq a + 0.5 rho Cd Af v^2 + m g Crr cos(theta) + m g sin(theta)
    P_wheel = F v
    P_bus = P_wheel/eta_trac                      (P_wheel >= 0)
          = regen_frac * eta_regen * P_wheel      (P_wheel <  0)
    P_bus = clip(P_bus, -P_regen_max, P_trac_max) + P_aux

Vehicle parameters: configs/nominal.yaml, section `vehicle` (Tesla Model 3, sourced; see the
source tags there). configs/reduced_storage.yaml keeps the reduced-storage plant of the sizing study.
Speed is interpolated with a shape-preserving PCHIP spline so acceleration is
continuous between 1 Hz samples.
"""
import os

import numpy as np
from scipy.interpolate import PchipInterpolator

G = 9.81


def load_speed(path):
    with open(path, encoding="utf-8-sig") as f:
        hdr = f.readline().strip().split(",")
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    if "cycSecs" in hdr:
        t, v = d[:, hdr.index("cycSecs")], d[:, hdr.index("cycMps")]
        g = d[:, hdr.index("cycGrade")] if "cycGrade" in hdr else np.zeros_like(t)
    else:
        t, v = d[:, hdr.index("time_s")], d[:, hdr.index("mps")]
        g = d[:, hdr.index("grade")] if "grade" in hdr else np.zeros_like(t)
    t = t - t[0]
    return t, np.maximum(v, 0.0), np.nan_to_num(g)


def power_table(t, v, grade, veh, dt=0.01):
    V = {k: float(val["value"]) for k, val in veh.items()}
    sp = PchipInterpolator(t, v)
    gp = np.interp
    tt = np.arange(0.0, t[-1], dt)
    vv = np.maximum(sp(tt), 0.0)
    aa = sp.derivative()(tt)
    th = np.arctan(gp(tt, t, grade))
    m = V["mass"]
    if "f0" in V:
        # measured coast-down road load F_rl = f0 + f1 v + f2 v^2 (includes rolling resistance and
        # aerodynamic drag), inertia with a rotational-mass factor, plus grade
        F = (m * V["rot_factor"] * aa + (V["f0"] * (vv > 0.05) + V["f1"] * vv + V["f2"] * vv ** 2)
             + m * G * np.sin(th))
    else:
        F = (m * (1 + V["rot_frac"]) * aa + 0.5 * V["rho_air"] * V["Cd"] * V["Af"] * vv ** 2
             + m * G * V["Crr"] * np.cos(th) * (vv > 0.05) + m * G * np.sin(th))
    Pw = F * vv
    Pb = np.where(Pw >= 0, Pw / V["eta_trac"], V["regen_frac"] * V["eta_regen"] * Pw)
    Pb = np.clip(Pb, -V["P_regen_max"], V["P_trac_max"]) + V["P_aux"]
    return tt, Pb, vv


def cycle_profile(name, cfg, root):
    path = os.path.join(root, "data", "cycles", name + ".csv")
    t, v, g = load_speed(path)
    tt, P, vv = power_table(t, v, g, cfg["vehicle"])
    return {"name": name, "t0": 0.0, "dt": float(tt[1] - tt[0]), "P": P, "speed": vv,
            "Tend": float(tt[-1]), "source": f"FASTSim cycle '{name}' + vehicle model of configs (vehicle section)"}


def scenario_profile(scen, dt=0.001):
    """Convert the analytic Fig.2 / Fig.7 reconstructions into the same table form."""
    tt = np.arange(0.0, scen["Tend"] + dt, dt)
    base = np.full_like(tt, scen["P0"])
    for ti, dP, tau in zip(scen["seg_t"], scen["seg_dP"], scen["seg_tau"]):
        base += dP * 0.5 * (1.0 + np.tanh((tt - ti) / tau))
    P = base * (1.0 + scen["rip_frac"] * np.sin(2 * np.pi * scen["rip_f"] * tt))
    return {"name": "paper_" + scen["name"], "t0": 0.0, "dt": dt, "P": P, "speed": None,
            "Tend": scen["Tend"], "events": scen["events"], "source": scen["source"]}
