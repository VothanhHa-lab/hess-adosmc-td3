"""Load-power profiles.

Both profiles are RECONSTRUCTED by eye from the manuscript figures (Fig. 2 and
Fig. 7); the manuscript gives no numeric definition. Levels/times are therefore
approximate and are listed here explicitly so they can be replaced by the
authors' original profile data.
P_L(t) = scale * [P0 + sum_i dP_i * 0.5*(1+tanh((t-t_i)/tau_i))] * (1 + rip_frac*sin(2 pi rip_f t))
"""
import numpy as np


def _from_levels(levels, tau):
    """levels: list of (t_switch, new_level_W). Returns increments."""
    t, dP = [], []
    prev = 0.0
    for ts, lvl in levels:
        t.append(ts)
        dP.append(lvl - prev)
        prev = lvl
    return np.array(t, float), np.array(dP, float), np.full(len(t), tau)


def transient():
    """Fig. 2: 0 kW -> ~60 kW at ~1 s (acceleration), -> ~-30 kW at ~4 s (regen),
    -> ~8 kW at ~5.5 s; small oscillation ~3 % at ~1.1 Hz."""
    levels = [(1.08, 60e3), (4.05, -30e3), (5.45, 8e3)]
    t, dP, tau = _from_levels(levels, 0.04)
    return dict(name="transient", Tend=6.0, P0=0.0, seg_t=t, seg_dP=dP, seg_tau=tau,
                rip_frac=0.03, rip_f=1.1,
                # event time = start of the tanh transition (centre - 2.5*tau, where the
                # transition has completed < 1 %) so each metric window contains its whole event
                events={"accel": 1.08 - 0.1, "regen": 4.05 - 0.1, "reaccel": 5.45 - 0.1},
                source="reconstructed from manuscript Fig. 2 (approximate)")


def urban():
    """Fig. 7: repeated acceleration / cruise / regenerative braking over 60 s."""
    levels = [(4.0, 38e3), (9.0, 19e3), (14.0, -24e3), (18.0, 0.0),
              (22.0, 52e3), (27.0, 25e3), (33.0, -32e3), (37.0, 7e3),
              (41.0, 50e3), (47.0, 21e3), (53.0, -29e3), (57.0, 0.0)]
    t, dP, tau = _from_levels(levels, 0.1)
    return dict(name="urban", Tend=60.0, P0=0.0, seg_t=t, seg_dP=dP, seg_tau=tau,
                rip_frac=0.03, rip_f=1.1,
                events={f"ev{i}": ts - 0.25 for i, (ts, _) in enumerate(levels)},
                source="reconstructed from manuscript Fig. 7 (approximate)")


SCENARIOS = {"transient": transient, "urban": urban}
