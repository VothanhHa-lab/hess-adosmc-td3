"""Minimal TD3 (Fujimoto, van Hoof & Meger, ICML 2018) in NumPy.

Two-hidden-layer MLPs with ReLU, manual back-propagation and Adam.
Written without a deep-learning framework so the study has no GPU/torch
dependency; correctness of the gradients is checked in tests/verify_agent.py
by finite differences.
"""
import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_act=None):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        if out_act == "tanh":   # small final layer for tanh policies
            self.W[-1] *= 0.1
        self.out_act = out_act
        self.m = [np.zeros_like(w) for w in self.W + self.b]
        self.v = [np.zeros_like(w) for w in self.W + self.b]
        self.t = 0

    def params(self):
        return self.W + self.b

    def forward(self, x):
        self.cache = [x]
        h = x
        L = len(self.W)
        for i in range(L):
            z = h @ self.W[i] + self.b[i]
            if i < L - 1:
                h = np.maximum(z, 0.0)
            else:
                h = np.tanh(z) if self.out_act == "tanh" else z
            self.cache.append(h)
        return h

    def backward(self, dout):
        """dout = dLoss/dOutput. Returns (grads list matching params(), dLoss/dInput)."""
        L = len(self.W)
        gW = [None] * L
        gb = [None] * L
        d = dout
        if self.out_act == "tanh":
            d = d * (1.0 - self.cache[-1] ** 2)
        for i in reversed(range(L)):
            hin = self.cache[i]
            gW[i] = hin.T @ d
            gb[i] = d.sum(axis=0)
            d = d @ self.W[i].T
            if i > 0:
                d = d * (self.cache[i] > 0.0)
        return gW + gb, d

    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        ps = self.params()
        for k, (p, g) in enumerate(zip(ps, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p[...] = q

    def soft_update(self, other, tau):
        for p, q in zip(self.params(), other.params()):
            p *= (1 - tau)
            p += tau * q

    def state(self):
        return [p.copy() for p in self.params()]

    def load(self, st):
        for p, q in zip(self.params(), st):
            p[...] = q


class Replay:
    def __init__(self, n, ds, da):
        self.s = np.zeros((n, ds)); self.a = np.zeros((n, da)); self.r = np.zeros(n)
        self.s2 = np.zeros((n, ds)); self.d = np.zeros(n)
        self.n, self.i, self.full = n, 0, False

    def add(self, s, a, r, s2, d):
        self.s[self.i] = s; self.a[self.i] = a; self.r[self.i] = r; self.s2[self.i] = s2; self.d[self.i] = d
        self.i = (self.i + 1) % self.n
        self.full = self.full or self.i == 0

    def size(self):
        return self.n if self.full else self.i

    def sample(self, rng, B):
        idx = rng.integers(0, self.size(), B)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class TD3:
    def __init__(self, ds, da, hp, seed):
        self.rng = np.random.default_rng(seed)
        H = hp["hidden"]
        self.actor = MLP([ds, H, H, da], self.rng, "tanh")
        self.actor_t = MLP([ds, H, H, da], self.rng, "tanh"); self.actor_t.copy_from(self.actor)
        self.q1 = MLP([ds + da, H, H, 1], self.rng); self.q2 = MLP([ds + da, H, H, 1], self.rng)
        self.q1_t = MLP([ds + da, H, H, 1], self.rng); self.q1_t.copy_from(self.q1)
        self.q2_t = MLP([ds + da, H, H, 1], self.rng); self.q2_t.copy_from(self.q2)
        self.hp = hp
        self.da = da
        self.it = 0
        self.buf = Replay(int(hp["buffer"]), ds, da)

    def act(self, s, noise=0.0):
        a = self.actor.forward(s[None, :])[0]
        if noise > 0:
            a = a + noise * self.rng.normal(size=self.da)
        return np.clip(a, -1.0, 1.0)

    def update(self):
        hp = self.hp
        s, a, r, s2, d = self.buf.sample(self.rng, int(hp["batch"]))
        self.it += 1
        na = self.actor_t.forward(s2)
        eps = np.clip(hp["policy_noise"] * self.rng.normal(size=na.shape), -hp["noise_clip"], hp["noise_clip"])
        na = np.clip(na + eps, -1, 1)
        x2 = np.concatenate([s2, na], 1)
        qt = np.minimum(self.q1_t.forward(x2), self.q2_t.forward(x2))[:, 0]
        y = r + hp["gamma"] * (1 - d) * qt
        x = np.concatenate([s, a], 1)
        losses = []
        for q in (self.q1, self.q2):
            qv = q.forward(x)[:, 0]
            err = qv - y
            losses.append(float(np.mean(err ** 2)))
            g, _ = q.backward((2.0 * err / len(err))[:, None])
            q.adam(g, hp["lr_critic"])
        aloss = None
        if self.it % int(hp["policy_delay"]) == 0:
            pa = self.actor.forward(s)
            xa = np.concatenate([s, pa], 1)
            qa = self.q1.forward(xa)
            aloss = float(-np.mean(qa))
            _, dx = self.q1.backward(-np.ones_like(qa) / len(qa))
            da = dx[:, s.shape[1]:]
            g, _ = self.actor.backward(da)
            self.actor.adam(g, hp["lr_actor"])
            self.actor_t.soft_update(self.actor, hp["tau"])
            self.q1_t.soft_update(self.q1, hp["tau"])
            self.q2_t.soft_update(self.q2, hp["tau"])
        return losses, aloss
