"""Chunked numba kernel for the two-tier study.

Same plant as kernel.py, but:
  * load power from a uniformly sampled table (linear interpolation),
  * simulation advances in chunks so a supervisory agent can change
    (taub, vsc_ref, lam) between chunks while all states persist,
  * allocation includes an SC energy-recovery term,
  * controller 3 = A-DO-SMC (adaptive switching gain + conditional integrator),
  * controller 4 = DO-ASMC baseline (same NDO and integrator, gain adapted from |s|, fixed layer).

Controller memory layout (mem):
  0 xi, 1 z, 2 Pb_f, 3 ib_int, 4 isc_int, 5 Dhat_prev, 6 dD_filt, 7 env, 8 e_last,
  9 ks_last, 10 phi_last, 11 K (DO-ASMC adaptive gain), 12 u_prev
Aux statistics (stats) accumulated per chunk (reset by caller):
  0 n, 1 sum e^2, 2 max|e|, 3 sum ib^2, 4 max|ib|, 5 sum slew^2, 6 max|slew|, 7 n_slew,
  8 n_sat, 9 min vsc, 10 max vsc, 11 sum |ib|*Ts, 12 sum ks, 13 sum isc^2,
  14 sum P* (measured v_dc * u; controller-internal load estimate for the MPC-EMS baseline),
  15 sum |du| (applied command increment per sample), 16 sum du^2
Block state for the 1 ms battery-current slew (blk): 0 sum, 1 count, 2 prev mean, 3 have prev
"""
import numpy as np
from numba import njit

from .kernel import box_qp, clip, sat1
from .params import (CDC, LB, LSC, RB, RSC, R0B, OCV_NOM, OCV_SLOPE, SOC_REF, QB, ETA_B,
                     CSC, RESR, IBMAX, ISCMAX, VSCMIN, VSCMAX, DMAX, VREF, LAM, KS, PHI,
                     CV, TAUB, WCI, KI_RATIO, ALLOC_MODE, KP_PI, KI_PI, CDC_NOM, LB_NOM, LSC_NOM,
                     TS, N_SUB, DELAY, CTRL, LOAD_SCALE, SIG_V, SIG_I,
                     VSC_REF, KE, PREC_MAX, K_MIN, BETA, TAU_F, TAU_ENV,
                     PHI0, GAMMA_A, EPS_S, PHI_FIX)

NX = 9
NMEM = 13
NSTAT = 17
LOG_COLS = ["t", "vdc", "iLb", "iLsc", "vb_t", "vC_sc", "soc_b", "P_L", "u", "u_un", "Dhat",
            "Dtrue", "ks", "phi", "taub", "vsc_ref", "lam", "iLb_ref", "iLsc_ref"]
NLOG = len(LOG_COLS)


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def table_load(t, t0, dt, P):
    x = (t - t0) / dt
    if x <= 0.0:
        return P[0]
    i = int(x)
    if i >= P.shape[0] - 1:
        return P[P.shape[0] - 1]
    f = x - i
    return P[i] * (1.0 - f) + P[i + 1] * f


@njit(cache=True)
def deriv_t(x, t, qb, qsc, p, t0, dtab, Ptab, dx):
    vdc = x[0]; iLb = x[1]; iLsc = x[2]; vC = x[3]; soc = x[4]
    ocv = p[OCV_NOM] + p[OCV_SLOPE] * (soc - p[SOC_REF])
    vb_t = ocv - p[R0B] * iLb
    vsc_t = vC - p[RESR] * iLsc
    PL = p[LOAD_SCALE] * table_load(t, t0, dtab, Ptab)
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
def rk4_t(x, t, h, qb, qsc, p, t0, dtab, Ptab, k1, k2, k3, k4, xt):
    n = x.shape[0]
    deriv_t(x, t, qb, qsc, p, t0, dtab, Ptab, k1)
    for i in range(n):
        xt[i] = x[i] + 0.5 * h * k1[i]
    deriv_t(xt, t + 0.5 * h, qb, qsc, p, t0, dtab, Ptab, k2)
    for i in range(n):
        xt[i] = x[i] + 0.5 * h * k2[i]
    deriv_t(xt, t + 0.5 * h, qb, qsc, p, t0, dtab, Ptab, k3)
    for i in range(n):
        xt[i] = x[i] + h * k3[i]
    deriv_t(xt, t + h, qb, qsc, p, t0, dtab, Ptab, k4)
    for i in range(n):
        x[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


@njit(cache=True)
def ctrl_step2(p, mem, vdc_m, iLb_m, iLsc_m, vb_m, vsc_m, H, Hinv, F, margin, dwork, lo, hi):
    Ts = p[TS]
    C = p[CDC_NOM]
    lam = p[LAM]
    ctrl = int(p[CTRL])
    e = vdc_m - p[VREF]
    xi = mem[0]
    kprime = p[KS] / p[PHI]           # boundary-layer slope (fixed: 240 1/s)

    dis_sc = p[ISCMAX] if vsc_m > p[VSCMIN] else 0.0
    ch_sc = p[ISCMAX] if vsc_m < p[VSCMAX] else 0.0
    umax = (vb_m * p[IBMAX] + vsc_m * dis_sc) / vdc_m
    umin = -(vb_m * p[IBMAX] + vsc_m * ch_sc) / vdc_m

    Dhat = 0.0
    ks = 0.0
    phi = p[PHI]
    integrate_ok = True
    if ctrl == 0:
        u_un = -C * (p[KP_PI] * e + p[KI_PI] * xi)
    elif ctrl == 1:
        Dhat = mem[1] + lam * e
        ks = p[KS]
        s = e + p[CV] * xi
        u_un = C * (-Dhat - p[CV] * e - ks * sat1(s / phi))
    elif ctrl == 2:
        Dhat = mem[1] + lam * e
        N = margin.shape[0]
        g = F[:, 0] * e + F[:, 1] * xi
        for i in range(N):
            lo[i] = umin / C + Dhat + margin[i]
            hi[i] = umax / C + Dhat - margin[i]
            if lo[i] > hi[i]:
                mid = 0.5 * (lo[i] + hi[i]); lo[i] = mid; hi[i] = mid
        box_qp(H, Hinv, g, lo, hi, dwork)
        d_unc = 0.0
        for j in range(N):
            d_unc -= Hinv[0, j] * g[j]
        u_un = C * (d_unc - Dhat)
    elif ctrl == 4:
        # DO-ASMC baseline: same NDO, sliding variable and conditional integrator; switching gain
        # adapted by the increase-decrease law dK/dt = gamma*|s|*sgn(|s| - eps_s), projected onto
        # K >= K_min; fixed boundary layer phi0.
        Dhat = mem[1] + lam * e
        K = mem[11] if mem[11] > 0.0 else p[K_MIN]
        phi = p[PHI0]
        ks = K
        s = e + p[CV] * xi
        u_un = C * (-Dhat - p[CV] * e - K * sat1(s / phi))
        if abs(s) > phi:
            integrate_ok = False
        a = abs(s)
        sg = 1.0 if a > p[EPS_S] else (-1.0 if a < p[EPS_S] else 0.0)
        K = K + Ts * p[GAMMA_A] * a * sg
        if K < p[K_MIN]:
            K = p[K_MIN]
        mem[11] = K
    else:
        # A-DO-SMC: |D~| estimated from the NDO itself (dDhat/dt = lam*D~), envelope,
        # ks = k_min + beta*env enforces Eq.(38) online; phi = ks/k' keeps the small-signal
        # poles {-cv, -k'}; conditional integrator (integrate only inside the layer).
        Dhat = mem[1] + lam * e
        dD = (Dhat - mem[5]) / Ts
        mem[5] = Dhat
        mem[6] += Ts / p[TAU_F] * (dD - mem[6])
        dtil = abs(mem[6]) / lam
        envd = mem[7] * np.exp(-Ts / p[TAU_ENV])
        mem[7] = dtil if dtil > envd else envd
        ks = p[K_MIN] + p[BETA] * mem[7]
        phi = ks / kprime if p[PHI_FIX] <= 0.0 else p[PHI_FIX]
        s = e + p[CV] * xi
        u_un = C * (-Dhat - p[CV] * e - ks * sat1(s / phi))
        if abs(s) > phi:
            integrate_ok = False
    if ctrl == 2:
        u = clip(C * (dwork[0] - Dhat), umin, umax)
    else:
        u = clip(u_un, umin, umax)
    mem[9] = ks
    mem[10] = phi

    if integrate_ok and (u == u_un or (u_un - u) * e > 0.0):
        mem[0] = xi + Ts * e
    if ctrl != 0:
        z = mem[1]
        mem[1] = z + Ts * (-lam * z - lam * (u / C + lam * e))
    mem[8] = e

    # allocation: LPF share + SC energy recovery, SC shortfall redistributed to battery
    Pstar = vdc_m * u
    mem[2] += Ts / p[TAUB] * (Pstar - mem[2])
    Prec = clip(p[KE] * 0.5 * p[CSC] * (p[VSC_REF] ** 2 - vsc_m ** 2), -p[PREC_MAX], p[PREC_MAX])
    Pb1 = clip(mem[2] + Prec, -vb_m * p[IBMAX], vb_m * p[IBMAX])
    Psc = clip(Pstar - Pb1, -vsc_m * ch_sc, vsc_m * dis_sc)
    Pb = clip(Pstar - Psc, -vb_m * p[IBMAX], vb_m * p[IBMAX])
    iLb_ref = Pb / vb_m
    iLsc_ref = Psc / vsc_m

    kpb = p[LB_NOM] * p[WCI]
    kib = kpb * p[WCI] / p[KI_RATIO]
    kps = p[LSC_NOM] * p[WCI]
    kis = kps * p[WCI] / p[KI_RATIO]
    eb = iLb_ref - iLb_m
    db_un = 1.0 - (vb_m - (kpb * eb + mem[3])) / vdc_m
    db = clip(db_un, 0.0, p[DMAX])
    if db == db_un:
        mem[3] += Ts * kib * eb
    es = iLsc_ref - iLsc_m
    ds_un = 1.0 - (vsc_m - (kps * es + mem[4])) / vdc_m
    dsc = clip(ds_un, 0.0, p[DMAX])
    if dsc == ds_un:
        mem[4] += Ts * kis * es
    return db, dsc, u, u_un, Dhat, ks, phi, iLb_ref, iLsc_ref


@njit(cache=True)
def run_chunk(p, x, mem, pend, blk, stats, k0, nsteps, t0tab, dtab, Ptab,
              H, Hinv, F, margin, log, ilog, log_every):
    """Advance nsteps controller samples starting at sample index k0.
    pend = [db_app, ds_app, db_pend, ds_pend].  Returns (ilog, status)."""
    Ts = p[TS]
    nsub = int(p[N_SUB])
    h = Ts / nsub
    Nh = margin.shape[0]
    dwork = np.zeros(Nh); lo = np.zeros(Nh); hi = np.zeros(Nh)
    k1 = np.zeros(NX); k2 = np.zeros(NX); k3 = np.zeros(NX); k4 = np.zeros(NX); xt = np.zeros(NX)
    blkn = max(1, int(np.round(1e-3 / Ts)))
    for kk in range(nsteps):
        k = k0 + kk
        t = k * Ts
        vdc = x[0]; iLb = x[1]; iLsc = x[2]
        if not (np.isfinite(vdc) and np.isfinite(iLb)) or vdc < 0.5 * p[VREF] or vdc > 1.5 * p[VREF]:
            return ilog, 1
        ocv = p[OCV_NOM] + p[OCV_SLOPE] * (x[4] - p[SOC_REF])
        vb_t = ocv - p[R0B] * iLb
        vsc_t = x[3] - p[RESR] * iLsc
        vdc_m = vdc + p[SIG_V] * np.random.normal()
        iLb_m = iLb + p[SIG_I] * np.random.normal()
        iLsc_m = iLsc + p[SIG_I] * np.random.normal()
        vb_m = vb_t + p[SIG_V] * np.random.normal()
        vsc_m = vsc_t + p[SIG_V] * np.random.normal()
        db, dsc, u, u_un, Dhat, ks, phi, ibr, iscr = ctrl_step2(
            p, mem, vdc_m, iLb_m, iLsc_m, vb_m, vsc_m, H, Hinv, F, margin, dwork, lo, hi)
        if p[DELAY] > 0.5:
            pend[0] = pend[2]; pend[1] = pend[3]; pend[2] = db; pend[3] = dsc
        else:
            pend[0] = db; pend[1] = dsc

        # statistics on TRUE signals
        e = vdc - p[VREF]
        stats[0] += 1.0
        stats[1] += e * e
        if abs(e) > stats[2]:
            stats[2] = abs(e)
        stats[3] += iLb * iLb
        if abs(iLb) > stats[4]:
            stats[4] = abs(iLb)
        if abs(u - u_un) > 1e-9:
            stats[8] += 1.0
        if x[3] < stats[9]:
            stats[9] = x[3]
        if x[3] > stats[10]:
            stats[10] = x[3]
        stats[11] += abs(iLb) * Ts
        stats[12] += ks
        stats[13] += iLsc * iLsc
        stats[14] += vdc_m * u
        if k > 0:
            du = u - mem[12]
            stats[15] += abs(du)
            stats[16] += du * du
        mem[12] = u
        blk[0] += iLb
        blk[1] += 1.0
        if blk[1] >= blkn:
            mean = blk[0] / blk[1]
            if blk[3] > 0.5:
                sl = (mean - blk[2]) / (blkn * Ts)
                stats[5] += sl * sl
                stats[7] += 1.0
                if abs(sl) > stats[6]:
                    stats[6] = abs(sl)
            blk[2] = mean; blk[3] = 1.0; blk[0] = 0.0; blk[1] = 0.0

        if log_every > 0 and k % log_every == 0 and ilog < log.shape[0]:
            PL = p[LOAD_SCALE] * table_load(t, t0tab, dtab, Ptab)
            iinj = (1.0 - pend[0]) * iLb + (1.0 - pend[1]) * iLsc
            log[ilog, 0] = t; log[ilog, 1] = vdc; log[ilog, 2] = iLb; log[ilog, 3] = iLsc
            log[ilog, 4] = vb_t; log[ilog, 5] = x[3]; log[ilog, 6] = x[4]; log[ilog, 7] = PL
            log[ilog, 8] = u; log[ilog, 9] = u_un; log[ilog, 10] = Dhat
            log[ilog, 11] = (iinj - PL / vdc) / p[CDC] - u / p[CDC_NOM]
            log[ilog, 12] = ks; log[ilog, 13] = phi; log[ilog, 14] = p[TAUB]
            log[ilog, 15] = p[VSC_REF]; log[ilog, 16] = p[LAM]
            log[ilog, 17] = ibr; log[ilog, 18] = iscr
            ilog += 1

        for j in range(nsub):
            rk4_t(x, t + j * h, h, pend[0], pend[1], p, t0tab, dtab, Ptab, k1, k2, k3, k4, xt)
    return ilog, 0


@njit(cache=True)
def bench_step2(p, M, H, Hinv, F, margin, seed):
    np.random.seed(seed)
    mem = np.zeros(NMEM)
    N = margin.shape[0]
    dwork = np.zeros(N); lo = np.zeros(N); hi = np.zeros(N)
    acc = 0.0
    for k in range(M):
        vdc = 400.0 + 5.0 * np.sin(2 * np.pi * k / 2000.0) + 0.2 * np.random.normal()
        out = ctrl_step2(p, mem, vdc, 50.0, 30.0, 296.0, 250.0, H, Hinv, F, margin, dwork, lo, hi)
        acc += out[0]
    return acc
