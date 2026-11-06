"""Computational-cost benchmark of one controller update (compiled numba code).

Measures the wall time of M consecutive calls to ctrl_step inside a compiled
loop (no Python call overhead) on the current machine.  This is a relative
indicator on a desktop CPU, NOT an embedded/DSP execution time.
"""
import time

import numpy as np
from numba import njit

from .kernel import ctrl_step


@njit(cache=True)
def _bench(p, M, H, Hinv, F, margin, vdc_amp, seed):
    np.random.seed(seed)
    mem = np.zeros(5)
    N = margin.shape[0]
    dwork = np.zeros(N); lo = np.zeros(N); hi = np.zeros(N)
    acc = 0.0
    nact = 0
    for k in range(M):
        vdc = 400.0 + vdc_amp * np.sin(2 * np.pi * k / 2000.0) + 0.2 * np.random.normal()
        out = ctrl_step(p, mem, vdc, 50.0, 30.0, 296.0, 250.0, H, Hinv, F, margin, dwork, lo, hi)
        acc += out[0]
        if out[10] > 0:
            nact += 1
    return acc, nact


def bench(p, mpc, M=200000, vdc_amp=5.0, repeats=7):
    _bench(p, 1000, mpc["H"], mpc["Hinv"], mpc["F"], mpc["margin"], vdc_amp, 0)  # compile/warm
    ts = []
    nact = 0
    for r in range(repeats):
        t0 = time.perf_counter()
        _, nact = _bench(p, M, mpc["H"], mpc["Hinv"], mpc["F"], mpc["margin"], vdc_amp, r)
        ts.append((time.perf_counter() - t0) / M)
    ts = np.array(ts) * 1e6
    return {"median_us": float(np.median(ts)), "min_us": float(ts.min()), "max_us": float(ts.max()),
            "M": M, "repeats": repeats, "qp_active_frac": nact / M}
