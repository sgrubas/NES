"""
Test problems for the hard-constraint prototypes: analytic velocity models, two of them with caustics.

Evaluation grid 241 x 241. Reference: closed form where it is the first arrival inside the box, otherwise the
2nd-order factored FMM on a 1921 x 1921 grid (every 8th node is an evaluation node).
Sources sit on the nodes of all FMM grids used here ((n - 1) in FMM_SIZES + [240, 1920]).

Every model is well-posed in its box: the fastest paths stay inside, so the in-box first arrival equals that of a
larger domain (checked with FMM on a 3x larger domain: identical to 1e-12; the closed forms match the in-box FMM to
its discretization error). A PINN has no boundary condition on the box, so when fastest paths leave it (e.g. Maxwell's
fish-eye with its focus inside the box) the in-box eikonal has several viscosity solutions and the test is ambiguous.
"""
import time
import numpy as np
import eikonalfm
from NES.velocity import BaseVelocity, VerticalGradient, LocAnomaly, MaxwellFishEye

N_EVAL = 241
N_REF = 1921
FMM_SIZES = [41, 81, 121, 241, 481, 961]
BOX = dict(xmin=[-1.2, -1.2], xmax=[1.2, 1.2])


class MultiGauss(BaseVelocity):
    """ v = v_bg + sum_k a_k exp(-|x - mu_k|^2 / (2 sigma_k^2)), with its analytic gradient """
    def __init__(self, v_bg, amps, mus, sigmas, xmin=None, xmax=None):
        self.v_bg, self.amps = float(v_bg), np.asarray(amps, float)
        self.mus, self.sig = np.asarray(mus, float), np.asarray(sigmas, float)
        self.xmin, self.xmax, self.dim = xmin, xmax, self.mus.shape[1]
        X = np.stack(np.meshgrid(*[np.linspace(a, b, 301) for a, b in zip(xmin, xmax)], indexing='ij'), -1)
        V = self(X)
        self.min, self.max = float(V.min()), float(V.max())

    def _g(self, X):
        d = np.asarray(X)[..., None, :] - self.mus
        return np.exp(-(d ** 2).sum(-1) / (2 * self.sig ** 2)), d

    def __call__(self, X):
        g, _ = self._g(X)
        return self.v_bg + (self.amps * g).sum(-1)

    def gradient(self, X):
        g, d = self._g(X)
        return (-(self.amps * g / self.sig ** 2)[..., None] * d).sum(-2)


# name: (title, velocity, source, closed form is the in-box first arrival, has a caustic)
MODELS = {
    'TwoGaussLow': ('Two low-velocity anomalies', MultiGauss(2.0, [-1.0, -1.0], [[-0.15, -0.4], [0.3, 0.45]],
                                                            [0.25, 0.25], **BOX), [-0.9, 0.0], False, True),
    'GaussLow': ('Gaussian low-velocity anomaly', LocAnomaly(2.0, 1.0, [0., 0.], [0.35, 0.35], **BOX),
                 [-0.9, 0.0], False, True),
    'GaussHigh': ('Gaussian high-velocity anomaly', LocAnomaly(2.0, 3.0, [0., 0.], [0.35, 0.35], **BOX),
                  [-0.9, 0.0], False, False),
    'HyperbolicLens': ('Hyperbolic lens', MaxwellFishEye(2.0, 1.0, high_velocity=True, xmin=[-0.6, -0.6],
                                                         xmax=[0.6, 0.6]), [-0.3, 0.15], True, False),
    'VerticalGradient': ('Vertical gradient', VerticalGradient(1.0, 2.0, xmin=[0., 0.], xmax=[1., 1.]),
                         [0.2, 0.2], True, False),
}


def grid(vel, n):
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(vel.xmin, vel.xmax)]
    return axes, np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)


def fmm(vel, xs, n):
    """ 2nd-order factored FMM on an n x n grid; returns (T, seconds) """
    axes, X = grid(vel, n)
    h = [a[1] - a[0] for a in axes]
    idx = tuple(int(round((c - a[0]) / hh)) for c, a, hh in zip(xs, axes, h))
    assert np.allclose([a[i] for a, i in zip(axes, idx)], xs, atol=1e-12), "source is not a grid node"
    V = vel(X)
    t0 = time.perf_counter()
    T = eikonalfm.factored_fast_marching(V, idx, h, 2) * eikonalfm.distance(V.shape, h, idx, indexing='ij')
    return T, time.perf_counter() - t0


def reference(name):
    """ Reference traveltimes on the evaluation grid and their kind """
    title, vel, xs, exact, _ = MODELS[name]
    xs = np.asarray(xs, float)
    if exact:
        return vel.time(grid(vel, N_EVAL)[1], xs), 'exact'
    T, _ = fmm(vel, xs, N_REF)
    k = (N_REF - 1) // (N_EVAL - 1)
    return T[::k, ::k], f'FMM {N_REF}x{N_REF}'


def fmm_benchmark(name, repeats=3):
    """ FMM error (RMAE vs the reference, at the FMM nodes) and wall time for each grid size """
    title, vel, xs, exact, _ = MODELS[name]
    xs = np.asarray(xs, float)
    T_fine = None if exact else fmm(vel, xs, N_REF)[0]
    rows = []
    for n in FMM_SIZES:
        runs = [fmm(vel, xs, n) for _ in range(repeats)]
        T = runs[0][0]
        if exact:
            ref = vel.time(grid(vel, n)[1], xs)
        else:
            k = (N_REF - 1) // (n - 1)
            ref = T_fine[::k, ::k]
        ok = ref > 0
        rows.append(dict(n=n, seconds=float(np.median([s for _, s in runs])),
                         rmae=float(np.abs(T - ref)[ok].sum() / ref[ok].sum()), values=n * n))
    return rows
