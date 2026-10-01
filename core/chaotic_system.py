# core/chaotic_system.py  (v2)
"""
6D hyperchaotic system of Wang et al. (2019), as used by Wu et al. (2024):

    dx/dt = a(y - x) + w - u - v
    dy/dt = c x - y - x z
    dz/dt = -b z + x y
    dw/dt = d w - y z
    du/dt = e v + z y
    dv/dt = r x

with a=10, b=8/3, c=28, d=-1, e=8, r=3 (hyperchaotic regime).

v2 changes (vs v1):
  * v1 used a different, Lorenz-like system whose largest Lyapunov exponent
    is ~0 for (10, 28, 8/3, 1, 1, 1) - i.e. not chaotic. Replaced by the
    system actually used in Wu et al. 2024 (Phys. Scr. 99, 045252, Eq. 1).
  * Fixed-step RK4 (h = 5e-3), deterministic; 10,000 warm-up steps (50 time
    units, ~e^38 amplification at lambda_1 ~ 0.77) so that a 1e-15 change of
    any initial value decorrelates the output (v1: 1000 steps of 1e-3 = 1 time
    unit; a 1e-15 perturbation left the key stream unchanged).
  * Integer / bit extraction uses the fractional-digit method
    floor(frac(|s| * 1e6) * m), which works for any sequence length
    (v1 min-max normalisation always returned 0 when n == 1, so the
    Hachimoji rule index was always 0).
"""
import hashlib
import numpy as np

PARAMS = dict(a=10.0, b=8.0 / 3.0, c=28.0, d=-1.0, e=8.0, r=3.0)
DT = 5e-3
WARMUP = 10000
SCALE = 1e6


class HyperchaoticSystem6D:
    def __init__(self, x0, y0, z0, w0, u0, v0, params=None, dt=DT, warmup=WARMUP):
        p = dict(PARAMS)
        if params:
            p.update(params)
        self.a, self.b, self.c = p["a"], p["b"], p["c"]
        self.d, self.e, self.r = p["d"], p["e"], p["r"]
        self.h = dt
        self.state = np.array([x0, y0, z0, w0, u0, v0], dtype=np.float64)
        self._buf = np.empty(0)
        for _ in range(warmup):
            self._rk4()

    def f(self, s):
        x, y, z, w, u, v = s
        return np.array([
            self.a * (y - x) + w - u - v,
            self.c * x - y - x * z,
            -self.b * z + x * y,
            self.d * w - y * z,
            self.e * v + z * y,
            self.r * x,
        ])

    def _rk4(self):
        h, s = self.h, self.state
        k1 = self.f(s)
        k2 = self.f(s + 0.5 * h * k1)
        k3 = self.f(s + 0.5 * h * k2)
        k4 = self.f(s + h * k3)
        self.state = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.state

    def raw(self, n):
        """n raw samples; all six state variables of each step are used."""
        while self._buf.size < n:
            steps = (n - self._buf.size) // 6 + 1
            out = np.empty((steps, 6))
            for i in range(steps):
                out[i] = self._rk4()
            self._buf = np.concatenate([self._buf, out.ravel()])
        seq, self._buf = self._buf[:n], self._buf[n:]
        return seq

    def ints(self, n, m):
        """n integers uniformly in [0, m)."""
        s = self.raw(n)
        frac = np.mod(np.abs(s) * SCALE, 1.0)
        return np.minimum((frac * m).astype(np.int64), m - 1)

    def bits(self, n):
        return self.ints(n, 2).astype(np.uint8)


def seed_from_digest(digest: bytes) -> dict:
    """Six initial values in (0, 1) from consecutive 40-bit chunks of a 256-bit digest."""
    assert len(digest) == 32
    vals = [(int.from_bytes(digest[5 * i:5 * i + 5], "big") + 1) / (2 ** 40 + 2) for i in range(6)]
    return dict(zip(["x0", "y0", "z0", "w0", "u0", "v0"], vals))


def lyapunov_spectrum(x0=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), n_steps=1_000_000, h=1e-3,
                      transient=20_000, reortho=10):
    """Benettin/QR estimate of the full Lyapunov spectrum."""
    p = PARAMS
    a, b, c, d, e, r = p["a"], p["b"], p["c"], p["d"], p["e"], p["r"]

    def F(s):
        x, y, z, w, u, v = s
        return np.array([a*(y-x)+w-u-v, c*x-y-x*z, -b*z+x*y, d*w-y*z, e*v+z*y, r*x])

    def J(s):
        x, y, z, w, u, v = s
        return np.array([[-a, a, 0, 1, -1, -1], [c-z, -1, -x, 0, 0, 0], [y, x, -b, 0, 0, 0],
                         [0, -z, -y, d, 0, 0], [0, z, y, 0, 0, e], [r, 0, 0, 0, 0, 0]])

    def step(s, Q):
        def g(s, Q):
            return F(s), J(s) @ Q
        k1 = g(s, Q); k2 = g(s+h/2*k1[0], Q+h/2*k1[1]); k3 = g(s+h/2*k2[0], Q+h/2*k2[1]); k4 = g(s+h*k3[0], Q+h*k3[1])
        return s+h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]), Q+h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])

    s, Q = np.array(x0, float), np.eye(6)
    for _ in range(transient):
        s, _ = step(s, np.zeros((6, 6)))
    le, T = np.zeros(6), 0.0
    for i in range(n_steps):
        s, Q = step(s, Q)
        if (i + 1) % reortho == 0:
            Q, R = np.linalg.qr(Q)
            le += np.log(np.abs(np.diag(R)))
            T += reortho * h
    return le / T
