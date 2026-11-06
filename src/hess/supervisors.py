"""Tier-2 supervisors: fixed, rule-based (SOC-adaptive LPF), TD3 agent with or
without the safety filter.  A supervisor maps an observation to the
supervisory parameter vector theta = (tau_b [s], vsc_ref [V], lambda [1/s]).

Safety filter (proposed):  theta_cmd = Pi_S(theta_prev, theta_agent)
  1) box   : tau in [0.05, 5] s (log), vsc_ref in [200, 290] V, lambda in [100, 800] 1/s
             (lambda*Ts <= 0.04 keeps the discretised NDO well inside its stability region)
  2) rate  : |d log tau| <= 0.2, |d vsc_ref| <= 5 V, |d lambda| <= 50 1/s per decision
             (slowly-varying parameters, needed by the stability argument of tier 1)
  3) energy reserve: vsc_ref within [v_lo, v_hi] so the SC can cover reserve_s seconds of the
             battery-limited power deficit / regenerative surplus
The tier-1 certificate (Eq. 38) is enforced inside A-DO-SMC for any lambda by the adaptive
switching gain, independently of the supervisor.
"""
import numpy as np


def _lerp(a, lo, hi):
    return lo + 0.5 * (a + 1.0) * (hi - lo)


def _loglerp(a, lo, hi):
    return float(np.exp(_lerp(a, np.log(lo), np.log(hi))))


class FixedSupervisor:
    name = "fixed"

    def __init__(self, cfg):
        self.theta = np.array([cfg["control"]["taub"]["value"], cfg["supervision"]["vsc_ref"]["value"],
                               cfg["control"]["lam"]["value"]], float)

    def reset(self):
        pass

    def decide(self, obs, info):
        return self.theta.copy(), None


class RuleSupervisor:
    """SOC-adaptive low-pass split (non-learning baseline, in the spirit of SOC-based
    adaptive filtering for HESS): high SC SOC -> long tau (SC takes more), low SOC -> short."""
    name = "rule"

    def __init__(self, cfg):
        self.cfg = cfg
        self.tmin, self.tmax = cfg["agent"]["rule_tau_range"]
        self.vref = cfg["supervision"]["vsc_ref"]["value"]
        self.lam = cfg["control"]["lam"]["value"]
        self.vlo = cfg["limits"]["vscmin"]["value"]; self.vhi = cfg["limits"]["vscmax"]["value"]

    def reset(self):
        pass

    def decide(self, obs, info):
        soc = (info["vsc"] - self.vlo) / (self.vhi - self.vlo)
        w = np.clip((soc - 0.2) / 0.6, 0.0, 1.0)
        return np.array([self.tmin + w * (self.tmax - self.tmin), self.vref, self.lam]), None


class SafetyFilter:
    def __init__(self, cfg, enabled=True, version=1):
        A = cfg["agent"]
        self.enabled = enabled
        self.version = version
        self.E_res = float(A.get("reserve_energy_J", 0.0))
        if enabled:
            self.tau_r, self.vsc_r, self.lam_r = A["tau_range"], A["vsc_range"], A["lam_range"]
        else:
            self.tau_r, self.vsc_r, self.lam_r = (A["unfiltered_tau_range"], A["unfiltered_vsc_range"],
                                                  A["unfiltered_lam_range"])
        self.rt, self.rv, self.rl = A["rate_log_tau"], A["rate_vsc"], A["rate_lam"]
        self.reserve = A["reserve_s"]
        V = cfg["vehicle"]; L = cfg["limits"]; P = cfg["plant"]
        self.Ptr = V["P_trac_max"]["value"]; self.Prg = V["P_regen_max"]["value"]
        self.Ib = L["Ibmax"]["value"]; self.vmin = L["vscmin"]["value"]; self.vmax = L["vscmax"]["value"]
        self.Csc = P["Csc"]["value"]
        self.Isc = L["Iscmax"]["value"]
        self.n_active = np.zeros(3, int)
        self.n = 0

    def map_action(self, a):
        return np.array([_loglerp(a[0], *self.tau_r), _lerp(a[1], *self.vsc_r), _lerp(a[2], *self.lam_r)])

    def __call__(self, theta_prev, a, vb):
        return self.project(theta_prev, self.map_action(a), vb)

    def project(self, theta_prev, th, vb):
        th = np.array(th, float)
        self.n += 1
        if not self.enabled:
            return th
        raw = th.copy()
        lt = np.clip(np.log(th[0]), np.log(theta_prev[0]) - self.rt, np.log(theta_prev[0]) + self.rt)
        th[0] = float(np.exp(lt))
        th[1] = np.clip(th[1], theta_prev[1] - self.rv, theta_prev[1] + self.rv)
        th[2] = np.clip(th[2], theta_prev[2] - self.rl, theta_prev[2] + self.rl)
        Pb = vb * self.Ib
        E_dis = self.reserve * max(self.Ptr - Pb, 0.0)
        E_chg = self.reserve * max(self.Prg - Pb, 0.0)
        v_lo = np.sqrt(self.vmin ** 2 + 2 * E_dis / self.Csc)
        v_hi = np.sqrt(max(self.vmax ** 2 - 2 * E_chg / self.Csc, 0.0))
        if self.version >= 2:
            # power capability: SC must still deliver P_trac_max - P_b,max after supplying the
            # largest train-set energy deficit E_res
            v_pow = max(self.Ptr - Pb, 0.0) / self.Isc
            v_lo = max(v_lo, np.sqrt(v_pow ** 2 + 2 * self.E_res / self.Csc))
        self.v_lo_last = v_lo
        th[1] = np.clip(th[1], max(v_lo, self.vsc_r[0]), min(v_hi, self.vsc_r[1]))
        th[0] = np.clip(th[0], *self.tau_r); th[2] = np.clip(th[2], *self.lam_r)
        self.n_active += (np.abs(th - raw) > 1e-9).astype(int)
        return th


class AgentSupervisor:
    def __init__(self, cfg, actor, filtered=True, name=None, version=1):
        self.actor = actor
        self.filter = SafetyFilter(cfg, filtered, version)
        self.name = name or ("td3+filter" if filtered else "td3-nofilter")
        self.theta0 = FixedSupervisor(cfg).theta.copy()

    def reset(self):
        self.prev = self.theta0.copy()

    def decide(self, obs, info, noise=0.0, rng=None):
        a = self.actor.forward(obs[None, :])[0]
        if noise > 0:
            a = np.clip(a + noise * rng.normal(size=a.shape), -1, 1)
        th = self.filter(self.prev, a, info["vb"])
        self.prev = th
        return th, a


class MPCEMSSupervisor:
    """Non-learning model-predictive energy-management baseline for tier 2.

    Decides the same supervisory variables as the agent (tau_b, vsc_ref) every dt_agent,
    under the SAME safety filter (v2: box, rate, power-capability reserve); lambda is kept
    at its nominal value because the energy model below does not depend on it.

    Prediction model (step dt = dt_agent, horizon N steps), identical in structure to the
    allocation layer of tier 1 and assuming tier 1 tracks P* = P_L:
        P_bf+ = P_bf + (1 - exp(-dt/tau)) (P_L - P_bf)
        P_rec = sat(kE 0.5 Csc (vref^2 - v^2), +-Prec_max)
        P_b1  = sat(P_bf + P_rec, +-vb Ib_max);  P_sc = sat(P_L - P_b1, SC limits)
        P_b   = sat(P_L - P_sc, +-vb Ib_max);    deficit = P_L - P_b - P_sc
        v+    = sqrt(v^2 - 2 P_sc dt / Csc)
    Stage cost = the agent's reward terms that the model can predict (battery RMS, battery
    slew, deficit as a saturation proxy, SC voltage-window penalty), same weights.
    Decision: enumeration of 5 x 5 candidate moves (d log tau, d vsc_ref) inside the filter's
    rate window, held over the horizon (move blocking), each projected by the filter.
    Load forecast:
        'persistence' : P_L(t+j) = mean P* = v_dc*u over the last decision interval (causal,
                        controller-internal signal only)
        'preview'     : true future P_L from the drive cycle (non-causal oracle; upper bound)
    """

    def __init__(self, cfg, forecast="persistence", horizon_s=None):
        self.cfg = cfg
        A = cfg["agent"]; S = cfg["supervision"]; P = cfg["plant"]; L = cfg["limits"]
        M = A.get("mpc_ems", {})
        self.forecast = forecast
        self.dt = float(A["dt_agent"])
        if horizon_s is None:
            h = M.get("horizon_s", 2.0)
            horizon_s = float(h[forecast]) if isinstance(h, dict) else float(h)
        hs = horizon_s
        self.N = max(1, int(round(hs / self.dt)))
        self.name = f"mpc-ems-{forecast}"
        self.filter = SafetyFilter(cfg, True, 2)
        self.theta0 = FixedSupervisor(cfg).theta.copy()
        self.kE = S["kE"]["value"]; self.Pr = S["Prec_max"]["value"]
        self.Csc = P["Csc"]["value"]; self.Ib = L["Ibmax"]["value"]; self.Isc = L["Iscmax"]["value"]
        self.vmin = L["vscmin"]["value"]; self.vmax = L["vscmax"]["value"]
        self.w = A["reward"]
        R = self.w
        self.pen_lo = float(R.get("sc_pen_lo", 190.0)); self.pen_hi = float(R.get("sc_pen_hi", 295.0))
        self.pen_lo_s = float(R.get("sc_pen_lo_scale", 10.0)); self.pen_hi_s = float(R.get("sc_pen_hi_scale", 2.0))
        self.deficit_ref = float(M.get("deficit_ref_W", 1000.0))
        n = int(M.get("n_grid", 5))
        g = np.linspace(-1.0, 1.0, n)
        G1, G2 = np.meshgrid(g, g, indexing="ij")
        self.dlt = G1.ravel() * A["rate_log_tau"]
        self.dv = G2.ravel() * A["rate_vsc"]
        self.solve_times = []

    def reset(self):
        self.prev = self.theta0.copy()

    def _cost(self, tau, vref, Pbf0, v0, vb, PL):
        w = self.w
        a = 1.0 - np.exp(-self.dt / tau)
        Pbf = np.full(tau.shape, Pbf0); v = np.full(tau.shape, v0)
        Pbmax = vb * self.Ib
        J = np.zeros(tau.shape); Pb_prev = None
        for j in range(PL.shape[0]):
            Pbf = Pbf + a * (PL[j] - Pbf)
            Prec = np.clip(self.kE * 0.5 * self.Csc * (vref ** 2 - v ** 2), -self.Pr, self.Pr)
            Pb1 = np.clip(Pbf + Prec, -Pbmax, Pbmax)
            dis = np.where(v > self.vmin, v * self.Isc, 0.0)
            chg = np.where(v < self.vmax, v * self.Isc, 0.0)
            Psc = np.clip(PL[j] - Pb1, -chg, dis)
            Pb = np.clip(PL[j] - Psc, -Pbmax, Pbmax)
            deficit = np.abs(PL[j] - Pb - Psc)
            v = np.sqrt(np.maximum(v ** 2 - 2.0 * Psc * self.dt / self.Csc, 1.0))
            ib = Pb / vb
            J += w["w_rms"] * (ib / self.Ib) ** 2
            if Pb_prev is not None:
                J += w["w_slew"] * ((ib - Pb_prev / vb) / self.dt / w["slew_ref"]) ** 2
            J += w["w_sat"] * np.minimum(deficit / self.deficit_ref, 1.0)
            J += w["w_soc"] * (np.maximum(self.pen_lo - v, 0.0) / self.pen_lo_s) ** 2
            J += w["w_soc"] * (np.maximum(v - self.pen_hi, 0.0) / self.pen_hi_s) ** 2
            Pb_prev = Pb
        return J / PL.shape[0]

    def decide(self, obs, info):
        import time
        t0 = time.perf_counter()
        vb = info["vb"]
        if self.forecast == "preview":
            PL = info["preview"](self.N, self.dt)
        else:
            PL = np.full(self.N, info["Pstar_mean"] if info["Pstar_mean"] != 0.0 else info["Pb_f"])
        cands = np.empty((self.dlt.size, 3))
        f = self.filter
        n0, a0 = f.n, f.n_active.copy()
        for i in range(self.dlt.size):
            th = np.array([self.prev[0] * np.exp(self.dlt[i]), self.prev[1] + self.dv[i], self.theta0[2]])
            cands[i] = f.project(self.prev, th, vb)
        f.n, f.n_active = n0, a0           # count only the applied decision below
        J = self._cost(cands[:, 0], cands[:, 1], info["Pb_f"], info["vsc"], vb, PL)
        best = cands[int(np.argmin(J))]
        th = f.project(self.prev, best, vb)
        self.prev = th
        self.solve_times.append(time.perf_counter() - t0)
        return th, None
