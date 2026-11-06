"""Episode runner / RL environment for the two-tier HESS controller.

One decision every dt_agent seconds; between decisions the compiled kernel
advances the fast loop (20 kHz controller, RK4 plant).  Observation (11-D,
normalised, uses only quantities available to the controller):
  0 RMS(e) last chunk / 2 V        1 D_hat / 7e4 V/s (load proxy)
  2 mean D_hat last 1 s / 7e4      3 mean D_hat last 5 s / 7e4
  4 2*SOC_sc-1 (Eq. 8)              5 i_b / 80 A
  6 log tau_b (norm)                7 vsc_ref (norm)          8 lambda (norm)
  9 saturation fraction last chunk 10 |D~| envelope / 5000 V/s
"""
from collections import deque

import numpy as np

from .design import design_mpc, dummy_mpc
from .kernel_sup import LOG_COLS, NLOG, NMEM, NSTAT, NX, run_chunk, seed_rng
from .params import CTRL_MPC, IDX, build_params

OBS_DIM = 11
ACT_DIM = 3


class Episode:
    def __init__(self, cfg, ctrl, profile, seed, t_start=0.0, duration=None, vsc0=None, soc0=None,
                 overrides=None, log_every=0, n_sub=5):
        self.cfg = cfg
        A = cfg["agent"]
        self.dt_agent = float(A["dt_agent"])
        ov = {"n_sub": n_sub}
        if vsc0 is not None:
            ov["vsc0"] = vsc0
        if soc0 is not None:
            ov["soc0"] = soc0
        ov.update(overrides or {})
        self.p = build_params(cfg, ctrl, overrides=ov)
        self.ctrl = ctrl
        if ctrl == CTRL_MPC:
            self.mpc = design_mpc(self.p, int(cfg["control"]["mpc_N"]["value"]),
                                  float(cfg["control"]["mpc_wmax"]["value"]))
        else:
            self.mpc = dummy_mpc()
        self.Ts = self.p[IDX["Ts"]]
        self.nchunk = int(round(self.dt_agent / self.Ts))
        # profile shifted so the episode starts at t=0 in kernel time
        self.t0tab = profile["t0"] - t_start
        self.dtab = profile["dt"]
        self.Ptab = np.ascontiguousarray(profile["P"], dtype=np.float64)
        Tend = profile["Tend"] - t_start
        self.duration = min(duration, Tend) if duration else Tend
        self.ndec = int(np.floor(self.duration / self.dt_agent + 1e-9))
        self.seed = int(seed)
        self.log_every = int(log_every)
        nlog = (self.ndec * self.nchunk) // self.log_every + 2 if self.log_every > 0 else 1
        self.log = np.full((nlog, NLOG), np.nan)
        self.ilog = 0
        R = A["reward"]
        self.w = R
        L = cfg["limits"]
        self.vsc_lo, self.vsc_hi = float(L["vscmin"]["value"]), float(L["vscmax"]["value"])
        self.ib_norm = float(L["Ibmax"]["value"])
        # SC-window penalty band (defaults of the reduced-storage plant: [190, 295] V, scales 10 V / 2 V)
        self.pen_lo = float(R.get("sc_pen_lo", 190.0)); self.pen_hi = float(R.get("sc_pen_hi", 295.0))
        self.pen_lo_s = float(R.get("sc_pen_lo_scale", 10.0)); self.pen_hi_s = float(R.get("sc_pen_hi_scale", 2.0))
        self.d_norm = float(R.get("dhat_norm", 7e4))
        self.reset()

    # --------------------------------------------------------------
    def reset(self):
        p = self.p
        self.x = np.zeros(NX)
        self.x[0] = p[IDX["vref"]]; self.x[3] = p[IDX["vsc0"]]; self.x[4] = p[IDX["soc0"]]
        self.mem = np.zeros(NMEM)
        ocv0 = p[IDX["ocv_nom"]] + p[IDX["ocv_slope"]] * (p[IDX["soc0"]] - p[IDX["soc_ref"]])
        db = 1.0 - ocv0 / p[IDX["vref"]]; ds = 1.0 - p[IDX["vsc0"]] / p[IDX["vref"]]
        self.pend = np.array([db, ds, db, ds])
        self.blk = np.zeros(4)
        self.k = 0
        self.dec = 0
        self.status = 0
        self.ilog = 0
        self.Dhist = deque(maxlen=50)
        self.last_stats = None
        self.theta = np.array([p[IDX["taub"]], p[IDX["vsc_ref"]], p[IDX["lam"]]])
        seed_rng(self.seed)
        self.tot = np.zeros(NSTAT); self.tot[9] = 1e9; self.tot[10] = -1e9
        self.decision_log = []
        return self.obs()

    def info(self):
        p = self.p
        ocv = p[IDX["ocv_nom"]] + p[IDX["ocv_slope"]] * (self.x[4] - p[IDX["soc_ref"]])
        st = self.last_stats
        Pst = st[14] / st[0] if st is not None and st[0] > 0 else 0.0
        return {"vsc": self.x[3], "vb": ocv - p[IDX["R0b"]] * self.x[1], "soc_b": self.x[4],
                "t": self.k * self.Ts, "Pstar_mean": Pst, "Pb_f": self.mem[2], "theta": self.theta.copy(),
                "preview": self.preview}

    def preview(self, n, dt):
        """Oracle load preview P_L(t + j*dt), j = 0..n-1 (used ONLY by the non-causal MPC-EMS bound)."""
        t = self.k * self.Ts + dt * np.arange(n)
        x = np.clip((t - self.t0tab) / self.dtab, 0.0, self.Ptab.shape[0] - 1.0)
        return self.p[IDX["load_scale"]] * np.interp(x, np.arange(self.Ptab.shape[0]), self.Ptab)

    def dhat(self):
        return self.mem[1] + self.p[IDX["lam"]] * self.mem[8]

    def obs(self):
        A = self.cfg["agent"]
        st = self.last_stats
        e_rms = np.sqrt(st[1] / st[0]) if st is not None else 0.0
        sat = st[8] / st[0] if st is not None else 0.0
        D = self.dhat()
        h = np.array(self.Dhist) if len(self.Dhist) else np.array([D])
        tr, vr, lr = A["tau_range"], A["vsc_range"], A["lam_range"]
        th = self.theta
        o = np.array([
            e_rms / 2.0, D / self.d_norm, h[-10:].mean() / self.d_norm, h.mean() / self.d_norm,
            2.0 * (self.x[3] - self.vsc_lo) / (self.vsc_hi - self.vsc_lo) - 1.0, self.x[1] / self.ib_norm,
            2.0 * (np.log(th[0]) - np.log(tr[0])) / (np.log(tr[1]) - np.log(tr[0])) - 1.0,
            2.0 * (th[1] - vr[0]) / (vr[1] - vr[0]) - 1.0,
            2.0 * (th[2] - lr[0]) / (lr[1] - lr[0]) - 1.0,
            sat, self.mem[7] / 5000.0])
        return np.clip(o, -5, 5)

    def apply_theta(self, th):
        p = self.p
        lam_old = p[IDX["lam"]]
        Dh = self.mem[1] + lam_old * self.mem[8]
        p[IDX["taub"]] = th[0]; p[IDX["vsc_ref"]] = th[1]; p[IDX["lam"]] = th[2]
        # bumpless observer-gain change: keep D_hat continuous
        self.mem[1] = Dh - th[2] * self.mem[8]
        self.theta = np.array(th, float)

    def reward(self, st):
        w = self.w
        n = max(st[0], 1.0)
        e2 = st[1] / n
        ib2 = st[3] / n
        sl2 = st[5] / max(st[7], 1.0)
        sat = st[8] / n
        pen = 0.0
        if st[9] < self.pen_lo:
            pen += ((self.pen_lo - st[9]) / self.pen_lo_s) ** 2
        if st[10] > self.pen_hi:
            pen += ((st[10] - self.pen_hi) / self.pen_hi_s) ** 2
        terms = np.array([w["w_v"] * e2 / w["e_ref"] ** 2, w["w_rms"] * ib2 / self.ib_norm ** 2,
                          w["w_slew"] * sl2 / w["slew_ref"] ** 2, w["w_sat"] * sat, w["w_soc"] * pen])
        return -min(float(terms.sum()), float(w.get("clip", 1e9))), terms

    def step(self, th):
        if th is not None:
            self.apply_theta(th)
        st = np.zeros(NSTAT); st[9] = 1e9; st[10] = -1e9
        self.ilog, self.status = run_chunk(self.p, self.x, self.mem, self.pend, self.blk, st, self.k,
                                           self.nchunk, self.t0tab, self.dtab, self.Ptab,
                                           self.mpc["H"], self.mpc["Hinv"], self.mpc["F"], self.mpc["margin"],
                                           self.log, self.ilog, self.log_every)
        self.k += self.nchunk
        self.dec += 1
        self.last_stats = st
        self.Dhist.append(self.dhat())
        # accumulate totals
        for i in (0, 1, 3, 5, 7, 8, 11, 12, 13, 14, 15, 16):
            self.tot[i] += st[i]
        for i in (2, 4, 6, 10):
            self.tot[i] = max(self.tot[i], st[i])
        self.tot[9] = min(self.tot[9], st[9])
        r, terms = self.reward(st)
        done = self.dec >= self.ndec or self.status != 0
        if self.status != 0:
            r -= 50.0
        self.decision_log.append([self.k * self.Ts, *self.theta, r])
        return self.obs(), r, done, terms

    def summary(self):
        T = self.tot
        n = max(T[0], 1)
        return {
            "status": "ok" if self.status == 0 else "FAILED(bus out of range)",
            "t_end_s": float(self.k * self.Ts),
            "max_dev_V": float(T[2]), "rms_e_V": float(np.sqrt(T[1] / n)),
            "ib_rms_A": float(np.sqrt(T[3] / n)), "ib_peak_A": float(T[4]),
            "ib_slew_rms_Aps": float(np.sqrt(T[5] / max(T[7], 1))), "ib_slew_max_Aps": float(T[6]),
            "ib_Ah": float(T[11] / 3600.0), "isc_rms_A": float(np.sqrt(T[13] / n)),
            "sat_frac": float(T[8] / n), "vsc_min_V": float(T[9]), "vsc_max_V": float(T[10]),
            "vsc_final_V": float(self.x[3]), "soc_b_final": float(self.x[4]),
            "ks_mean_Vps": float(T[12] / n),
            "du_rms_A": float(np.sqrt(T[16] / n)), "du_tv_A": float(T[15]),
            "constraint_violation": bool(T[4] > 1.01 * self.p[IDX["Ibmax"]]),
        }

    def log_dict(self):
        L = self.log[: self.ilog]
        return {c: L[:, i] for i, c in enumerate(LOG_COLS)}


def run_episode(cfg, ctrl, profile, supervisor, seed, log_every=0, duration=None, t_start=0.0,
                vsc0=None, soc0=None, overrides=None, n_sub=5):
    ep = Episode(cfg, ctrl, profile, seed, t_start=t_start, duration=duration, vsc0=vsc0, soc0=soc0,
                 overrides=overrides, log_every=log_every, n_sub=n_sub)
    supervisor.reset()
    obs = ep.obs()
    R = 0.0
    done = ep.ndec == 0
    while not done:
        th, _ = supervisor.decide(obs, ep.info())
        obs, r, done, _ = ep.step(th)
        R += r
    s = ep.summary()
    s["return"] = R
    s["n_decisions"] = ep.dec
    return ep, s
