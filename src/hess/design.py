"""Offline controller design: robust (constraint-tightened) MPC matrices.

Outer-loop prediction model (discrete, sampling Ts), state x = [e_v, xi], xi = int e_v dt:
    e_{k+1}  = e_k + Ts * delta_k              delta = u/C_nom + D_hat   [V/s]
    xi_{k+1} = xi_k + Ts * e_k
The NDO estimate D_hat is held constant over the horizon (offset-free MPC).
Weights Q = diag(qe, qxi), r = 1 are fitted so that the unconstrained MPC
(= discrete LQR, terminal cost from the DARE) has closed-loop poles close to
exp(-cv*Ts), exp(-(ks/phi)*Ts), i.e. the same small-signal poles as DO-SMC
inside its boundary layer and as the PI baseline.  Constraint tightening
follows the restricted-constraint idea of Chisci, Rossiter & Zappa (2001):
stage-i input bounds are shrunk by sum_{j<i} |K A_K^j B| * w_max.
"""
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

from .params import IDX


def _lqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def design_mpc(p, N, wmax):
    Ts = p[IDX["Ts"]]
    cv = p[IDX["cv"]]
    kprime = p[IDX["ks"]] / p[IDX["phi"]]
    A = np.array([[1.0, 0.0], [Ts, 1.0]])
    B = np.array([[Ts], [0.0]])
    R = np.array([[1.0]])
    target = np.sort(np.exp(-np.array([cv, kprime]) * Ts))

    def cost(logq):
        Q = np.diag(np.exp(logq))
        K, _ = _lqr(A, B, Q, R)
        ev = np.sort(np.real(np.linalg.eigvals(A - B @ K)))
        # compare continuous-equivalent rates (1/s)
        return float(np.sum((np.log(ev) / Ts - np.log(target) / Ts) ** 2))

    # initial guess from continuous LQR intuition
    best = None
    for g0 in ([np.log(kprime**2 * cv), np.log((kprime * cv) ** 2)],
               [20.0, 30.0], [15.0, 25.0], [25.0, 35.0]):
        res = minimize(cost, np.array(g0, float), method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    Q = np.diag(np.exp(best.x))
    K, P = _lqr(A, B, Q, R)
    AK = A - B @ K
    poles_z = np.linalg.eigvals(AK)
    poles_s = np.log(poles_z.astype(complex)) / Ts

    # condensed prediction x_1..x_N
    n, m = 2, 1
    Phi = np.zeros((n * N, n))
    Gam = np.zeros((n * N, m * N))
    Ai = np.eye(n)
    for i in range(N):
        Ai = A @ Ai
        Phi[i * n:(i + 1) * n, :] = Ai
        for j in range(i + 1):
            Gam[i * n:(i + 1) * n, j:j + 1] = np.linalg.matrix_power(A, i - j) @ B
    Qbar = np.kron(np.eye(N), Q)
    Qbar[-n:, -n:] = P
    H = Gam.T @ Qbar @ Gam + np.kron(np.eye(N), R)
    F = Gam.T @ Qbar @ Phi
    Hinv = np.linalg.inv(H)

    # stage-wise tightening margins (in delta units, V/s)
    margin = np.zeros(N)
    acc = 0.0
    M = np.eye(n)
    for i in range(N):
        margin[i] = acc
        acc += abs((K @ M @ B).item()) * wmax
        M = AK @ M

    info = {
        "Q_diag": np.diag(Q).tolist(), "R": 1.0, "K": K.ravel().tolist(),
        "fit_cost": float(best.fun),
        "target_poles_s": (-np.array([cv, kprime])).tolist(),
        "achieved_poles_s": [complex(z).real for z in poles_s],
        "achieved_poles_s_imag": [complex(z).imag for z in poles_s],
        "margin_last_stage_Vps": float(margin[-1]),
        "margin_last_stage_A": float(margin[-1] * p[IDX["Cdc_nom"]]),
        "N": N, "wmax_Vps": wmax,
        "cond_H": float(np.linalg.cond(H)),
    }
    return {"H": H, "Hinv": Hinv, "F": F, "margin": margin, "K": K.ravel(), "info": info}


def dummy_mpc(N=2):
    """Placeholder arrays for non-MPC runs (numba needs consistent types)."""
    return {"H": np.eye(N), "Hinv": np.eye(N), "F": np.zeros((N, 2)), "margin": np.zeros(N),
            "K": np.zeros(2), "info": {}}
