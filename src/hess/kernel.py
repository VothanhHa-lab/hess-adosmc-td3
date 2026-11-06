"""Numba simulation kernel: battery-SC HESS + PI / DO-SMC / robust MPC.

Plant (true parameters, averaged or switched bidirectional boost legs):
    Lb  diLb/dt  = vb_t  - rb  iLb  - (1-qb)  vdc          vb_t  = OCV(SOC) - R0b iLb
    Lsc diLsc/dt = vsc_t - rsc iLsc - (1-qsc) vdc          vsc_t = vC - Resr iLsc
    Cdc dvdc/dt  = (1-qb) iLb + (1-qsc) iLsc - P_L(t)/vdc  (constant-power traction load, Eq.(2))
    Csc dvC/dt   = -iLsc                                   (Eq.(8))
    dSOC/dt      = -eta_b iLb / (3600 Qb)                  (Eq.(7))
q = duty d (averaged) or switch state from a triangular carrier (switched).
NOTE: the manuscript writes Eqs.(3)-(4) as L di/dt = u v_src - v_dc - r i, which with
v_src < v_dc would need u > 1 and is not a duty ratio; the standard averaged boost form
above is used instead (documented in README).

Extra integrated energy states give an exact energy-balance check.
"""
import numpy as np
from numba import njit

from .params import (CDC, LB, LSC, RB, RSC, R0B, OCV_NOM, OCV_SLOPE, SOC_REF, SOC0, QB, ETA_B,
                     CSC, RESR, VSC0, IBMAX, ISCMAX, VSCMIN, VSCMAX, DMAX, VREF, LAM, KS, PHI,
                     CV, TAUB, WCI, KI_RATIO, ALLOC_MODE, KP_PI, KI_PI, CDC_NOM, LB_NOM, LSC_NOM,
                     TS, N_SUB, SWITCHED, DELAY, CTRL, LOAD_SCALE, SIG_V, SIG_I)

NX = 9          # vdc, iLb, iLsc, vC, soc, E_b, E_sc, E_r, E_load
NLOG = 25
LOG_COLS = ["t", "vdc", "iLb", "iLsc", "vb_t", "vC_sc", "soc_b", "P_L", "i_load", "i_inj",
            "u", "u_un", "Dhat", "Dtrue", "s", "iLb_ref", "iLsc_ref", "db", "dsc", "umax",
            "umin", "qp_iter", "rip_b_pp", "rip_sc_pp", "e_meas"]


@njit(cache=True)
def load_power(t, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, scale):
    base = P0
    for i in range(seg_t.shape[0]):
        base += seg_dP[i] * 0.5 * (1.0 + np.tanh((t - seg_t[i]) / seg_tau[i]))
    return scale * base * (1.0 + rip_frac * np.sin(2.0 * np.pi * rip_f * t))


@njit(cache=True)
def deriv(x, t, qb, qsc, p, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, dx):
    vdc = x[0]; iLb = x[1]; iLsc = x[2]; vC = x[3]; soc = x[4]
    ocv = p[OCV_NOM] + p[OCV_SLOPE] * (soc - p[SOC_REF])
    vb_t = ocv - p[R0B] * iLb
    vsc_t = vC - p[RESR] * iLsc
    PL = load_power(t, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, p[LOAD_SCALE])
    iload = PL / vdc
    dx[0] = ((1.0 - qb) * iLb + (1.0 - qsc) * iLsc - iload) / p[CDC]
    dx[1] = (vb_t - p[RB] * iLb - (1.0 - qb) * vdc) / p[LB]
    dx[2] = (vsc_t - p[RSC] * iLsc - (1.0 - qsc) * vdc) / p[LSC]
    dx[3] = -iLsc / p[CSC]
    dx[4] = -p[ETA_B] * iLb / (3600.0 * p[QB])
    dx[5] = vb_t * iLb
    dx[6] = vsc_t * iLsc
    dx[7] = p[RB] * iLb * iLb + p[RSC] * iLsc * iLsc
    dx[8] = PL


@njit(cache=True)
def rk4(x, t, h, qb, qsc, p, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, k1, k2, k3, k4, xt):
    n = x.shape[0]
    deriv(x, t, qb, qsc, p, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, k1)
    for i in range(n):
        xt[i] = x[i] + 0.5 * h * k1[i]
    deriv(xt, t + 0.5 * h, qb, qsc, p, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, k2)
    for i in range(n):
        xt[i] = x[i] + 0.5 * h * k2[i]
    deriv(xt, t + 0.5 * h, qb, qsc, p, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, k3)
    for i in range(n):
        xt[i] = x[i] + h * k3[i]
    deriv(xt, t + h, qb, qsc, p, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, k4)
    for i in range(n):
        x[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


@njit(cache=True)
def clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=True)
def sat1(x):
    return clip(x, -1.0, 1.0)


@njit(cache=True)
def box_qp(H, Hinv, g, lo, hi, x):
    """min 0.5 x'Hx + g'x  s.t. lo <= x <= hi.
    Unconstrained solution if feasible, else primal-dual active set
    (Hintermueller-Ito-Kunisch).  Returns (iterations, KKT residual)."""
    N = g.shape[0]
    x[:] = -Hinv @ g
    ok = True
    for i in range(N):
        if x[i] < lo[i] - 1e-12 or x[i] > hi[i] + 1e-12:
            ok = False
            break
    if ok:
        return 0, 0.0
    mu = np.zeros(N)
    for i in range(N):
        x[i] = clip(x[i], lo[i], hi[i])
    cpar = 0.0
    for i in range(N):
        cpar += H[i, i]
    cpar /= N
    st = np.zeros(N, np.int64)
    prev = np.full(N, 9, np.int64)
    it = 0
    for it in range(1, 60):
        for i in range(N):
            if mu[i] + cpar * (x[i] - hi[i]) > 0.0:
                st[i] = 1
            elif mu[i] + cpar * (x[i] - lo[i]) < 0.0:
                st[i] = -1
            else:
                st[i] = 0
        same = True
        for i in range(N):
            if st[i] != prev[i]:
                same = False
        if same:
            break
        prev[:] = st
        nI = 0
        for i in range(N):
            if st[i] == 1:
                x[i] = hi[i]
            elif st[i] == -1:
                x[i] = lo[i]
            else:
                nI += 1
        if nI > 0:
            I = np.empty(nI, np.int64)
            k = 0
            for i in range(N):
                if st[i] == 0:
                    I[k] = i
                    k += 1
            HII = np.empty((nI, nI))
            rhs = np.empty(nI)
            for a in range(nI):
                ia = I[a]
                r = -g[ia]
                for j in range(N):
                    if st[j] != 0:
                        r -= H[ia, j] * x[j]
                rhs[a] = r
                for b in range(nI):
                    HII[a, b] = H[ia, I[b]]
            xI = np.linalg.solve(HII, rhs)
            for a in range(nI):
                x[I[a]] = xI[a]
        grad = H @ x + g
        for i in range(N):
            mu[i] = -grad[i] if st[i] != 0 else 0.0
    # KKT residual: stationarity on free set + primal feasibility + dual sign
    grad = H @ x + g
    res = 0.0
    for i in range(N):
        if x[i] > lo[i] + 1e-9 and x[i] < hi[i] - 1e-9:
            res = max(res, abs(grad[i]))
        elif x[i] >= hi[i] - 1e-9:
            res = max(res, max(grad[i], 0.0))     # at upper bound need grad <= 0
        else:
            res = max(res, max(-grad[i], 0.0))    # at lower bound need grad >= 0
    return it, res


@njit(cache=True)
def ctrl_step(p, mem, vdc_m, iLb_m, iLsc_m, vb_m, vsc_m, H, Hinv, F, margin, dwork, lo, hi):
    """One sampling period of the full digital controller.
    mem: [xi, z, Pb_f, ib_int, isc_int].  Returns duties and diagnostics."""
    Ts = p[TS]
    C = p[CDC_NOM]
    lam = p[LAM]
    ctrl = int(p[CTRL])
    e = vdc_m - p[VREF]
    xi = mem[0]

    # HESS total-current (bus-side) feasibility limits from measured terminal voltages
    dis_sc = p[ISCMAX] if vsc_m > p[VSCMIN] else 0.0
    ch_sc = p[ISCMAX] if vsc_m < p[VSCMAX] else 0.0
    umax = (vb_m * p[IBMAX] + vsc_m * dis_sc) / vdc_m
    umin = -(vb_m * p[IBMAX] + vsc_m * ch_sc) / vdc_m

    Dhat = 0.0
    s = 0.0
    qp_it = 0
    if ctrl == 0:                         # PI (same small-signal poles as DO-SMC)
        u_un = -C * (p[KP_PI] * e + p[KI_PI] * xi)
        u = clip(u_un, umin, umax)
    elif ctrl == 1:                       # DO-SMC, Eqs.(13),(18),(20)
        Dhat = mem[1] + lam * e
        s = e + p[CV] * xi
        u_un = C * (-Dhat - p[CV] * e - p[KS] * sat1(s / p[PHI]))
        u = clip(u_un, umin, umax)
    else:                                 # constraint-tightened offset-free MPC
        Dhat = mem[1] + lam * e
        N = margin.shape[0]
        g = F[:, 0] * e + F[:, 1] * xi
        for i in range(N):
            lo[i] = umin / C + Dhat + margin[i]
            hi[i] = umax / C + Dhat - margin[i]
            if lo[i] > hi[i]:
                mid = 0.5 * (lo[i] + hi[i])
                lo[i] = mid
                hi[i] = mid
        qp_it, _res = box_qp(H, Hinv, g, lo, hi, dwork)
        d_unc = 0.0
        for j in range(N):
            d_unc -= Hinv[0, j] * g[j]
        u_un = C * (d_unc - Dhat)
        u = clip(C * (dwork[0] - Dhat), umin, umax)

    # conditional-integration anti-windup (identical rule for all controllers)
    if u == u_un or (u_un - u) * e > 0.0:
        mem[0] = xi + Ts * e
    # NDO auxiliary state, Eq.(14) with f = 0, g = 1/C_nom (forward Euler)
    if ctrl != 0:
        z = mem[1]
        mem[1] = z + Ts * (-lam * z - lam * (u / C + lam * e))

    # battery / SC allocation, Eqs.(23)-(27) in power form
    Pstar = vdc_m * u
    mem[2] += Ts / p[TAUB] * (Pstar - mem[2])
    if p[ALLOC_MODE] < 0.5:              # paper-literal: i_sc* = i_HESS* - i_b*(unsaturated)
        Pb_un = mem[2]
        iLb_ref = clip(Pb_un / vb_m, -p[IBMAX], p[IBMAX])
        iLsc_ref = clip((Pstar - Pb_un) / vsc_m, -ch_sc, dis_sc)
    else:                                # SC shortfall redistributed to battery
        Pb1 = clip(mem[2], -vb_m * p[IBMAX], vb_m * p[IBMAX])
        Psc = clip(Pstar - Pb1, -vsc_m * ch_sc, vsc_m * dis_sc)
        Pb = clip(Pstar - Psc, -vb_m * p[IBMAX], vb_m * p[IBMAX])
        iLb_ref = Pb / vb_m
        iLsc_ref = Psc / vsc_m

    # inner PI current loops with voltage feed-forward -> duty
    kpb = p[LB_NOM] * p[WCI]
    kib = kpb * p[WCI] / p[KI_RATIO]
    kps = p[LSC_NOM] * p[WCI]
    kis = kps * p[WCI] / p[KI_RATIO]
    eb = iLb_ref - iLb_m
    vLb = kpb * eb + mem[3]
    db_un = 1.0 - (vb_m - vLb) / vdc_m
    db = clip(db_un, 0.0, p[DMAX])
    if db == db_un:
        mem[3] += Ts * kib * eb
    es = iLsc_ref - iLsc_m
    vLs = kps * es + mem[4]
    ds_un = 1.0 - (vsc_m - vLs) / vdc_m
    dsc = clip(ds_un, 0.0, p[DMAX])
    if dsc == ds_un:
        mem[4] += Ts * kis * es
    return db, dsc, u, u_un, Dhat, s, iLb_ref, iLsc_ref, umax, umin, qp_it


@njit(cache=True)
def simulate(p, Tend, log_every, seed, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f,
             H, Hinv, F, margin):
    np.random.seed(seed)
    Ts = p[TS]
    nsub = int(p[N_SUB])
    h = Ts / nsub
    nT = int(np.round(Tend / Ts))
    nlog = nT // log_every + 1
    log = np.full((nlog, NLOG), np.nan)
    Nh = margin.shape[0]
    dwork = np.zeros(Nh); lo = np.zeros(Nh); hi = np.zeros(Nh)

    x = np.zeros(NX)
    x[0] = p[VREF]; x[3] = p[VSC0]; x[4] = p[SOC0]
    k1 = np.zeros(NX); k2 = np.zeros(NX); k3 = np.zeros(NX); k4 = np.zeros(NX); xt = np.zeros(NX)
    mem = np.zeros(5)
    ocv0 = p[OCV_NOM] + p[OCV_SLOPE] * (p[SOC0] - p[SOC_REF])
    db_app = 1.0 - ocv0 / p[VREF]
    ds_app = 1.0 - p[VSC0] / p[VREF]
    db_pend = db_app; ds_pend = ds_app
    status = 0
    ilog = 0
    rip_b = 0.0; rip_s = 0.0
    for k in range(nT):
        t = k * Ts
        vdc = x[0]; iLb = x[1]; iLsc = x[2]
        if not (np.isfinite(vdc) and np.isfinite(iLb) and np.isfinite(iLsc)) or vdc < 0.5 * p[VREF] or vdc > 1.5 * p[VREF]:
            status = 1
            break
        ocv = p[OCV_NOM] + p[OCV_SLOPE] * (x[4] - p[SOC_REF])
        vb_t = ocv - p[R0B] * iLb
        vsc_t = x[3] - p[RESR] * iLsc
        # sampled measurements (sampling at carrier peak -> period-average in switched mode)
        vdc_m = vdc + p[SIG_V] * np.random.normal()
        iLb_m = iLb + p[SIG_I] * np.random.normal()
        iLsc_m = iLsc + p[SIG_I] * np.random.normal()
        vb_m = vb_t + p[SIG_V] * np.random.normal()
        vsc_m = vsc_t + p[SIG_V] * np.random.normal()

        db, dsc, u, u_un, Dhat, s, iLb_ref, iLsc_ref, umax, umin, qp_it = ctrl_step(
            p, mem, vdc_m, iLb_m, iLsc_m, vb_m, vsc_m, H, Hinv, F, margin, dwork, lo, hi)
        if p[DELAY] > 0.5:
            db_app = db_pend; ds_app = ds_pend
            db_pend = db; ds_pend = dsc
        else:
            db_app = db; ds_app = dsc

        if k % log_every == 0 and ilog < nlog:
            PL = load_power(t, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, p[LOAD_SCALE])
            iload = PL / vdc
            iinj = (1.0 - db_app) * iLb + (1.0 - ds_app) * iLsc
            edot_true = (iinj - iload) / p[CDC]
            log[ilog, 0] = t; log[ilog, 1] = vdc; log[ilog, 2] = iLb; log[ilog, 3] = iLsc
            log[ilog, 4] = vb_t; log[ilog, 5] = x[3]; log[ilog, 6] = x[4]; log[ilog, 7] = PL
            log[ilog, 8] = iload; log[ilog, 9] = iinj; log[ilog, 10] = u; log[ilog, 11] = u_un
            log[ilog, 12] = Dhat; log[ilog, 13] = edot_true - u / p[CDC_NOM]; log[ilog, 14] = s
            log[ilog, 15] = iLb_ref; log[ilog, 16] = iLsc_ref; log[ilog, 17] = db_app
            log[ilog, 18] = ds_app; log[ilog, 19] = umax; log[ilog, 20] = umin
            log[ilog, 21] = qp_it; log[ilog, 22] = rip_b; log[ilog, 23] = rip_s
            log[ilog, 24] = vdc_m - p[VREF]
            ilog += 1

        # integrate plant over one sampling period
        bmin = 1e9; bmax = -1e9; smin = 1e9; smax = -1e9
        for j in range(nsub):
            tj = t + j * h
            if p[SWITCHED] > 0.5:
                c = abs(1.0 - 2.0 * (j + 0.5) / nsub)   # triangle, peak at sampling instant
                qb = 1.0 if c < db_app else 0.0
                qs = 1.0 if c < ds_app else 0.0
            else:
                qb = db_app; qs = ds_app
            rk4(x, tj, h, qb, qs, p, P0, seg_t, seg_dP, seg_tau, rip_frac, rip_f, k1, k2, k3, k4, xt)
            if x[1] < bmin: bmin = x[1]
            if x[1] > bmax: bmax = x[1]
            if x[2] < smin: smin = x[2]
            if x[2] > smax: smax = x[2]
        rip_b = bmax - bmin
        rip_s = smax - smin
    return log[:ilog], status, x
