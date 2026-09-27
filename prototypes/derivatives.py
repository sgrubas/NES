"""
Accuracy of traveltime derivatives: NES (automatic differentiation of a smooth network) against FMM (finite
differences of a grid solution), on models with closed-form traveltimes.

Exact T, grad T and hess T come from automatic differentiation of the closed forms. Errors are relative L1 norms,
    e_T = sum|T - T*| / sum|T*|,   e_G = sum|G - G*| / sum|G*|,   e_L = sum|lap - lap*| / sum|lap*|,
    e_H = sum||H - H*||_F / sum||H*||_F,
in two zones: far field (R >= 0.1) and near the source (0.02 <= R < 0.1), where T has a cone singularity and the
laplacian grows like 1/R.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

import problems as P
import hard_nes as H

jax.config.update("jax_enable_x64", True)

ZONES = {'far': (0.1, np.inf), 'near': (0.02, 0.1)}


def closed_form(name):
    """ Closed-form traveltime T(x) for the source of `name`, in JAX """
    title, vel, xs, exact, _ = P.MODELS[name]
    assert exact, f'{name} has no closed form'
    xs = jnp.asarray(xs, dtype=float)
    if name == 'VerticalGradient':
        v0, a = float(vel.v0), float(vel.a)

        def T(x):
            vs, vx = v0 + a * xs[-1], v0 + a * x[-1]
            return jnp.arccosh(1.0 + a * a * jnp.sum((x - xs) ** 2) / (2 * vs * vx)) / a
    else:                                                     # MaxwellFishEye / hyperbolic lens
        v0, Rl, k, c = float(vel.v0), float(vel.R), float(vel.k), jnp.asarray(vel.center, dtype=float)

        def T(x):
            u1, u2 = (xs - c) / Rl, (x - c) / Rl
            q = jnp.sqrt(jnp.sum((u2 - u1) ** 2) / ((1 + k * jnp.sum(u1 ** 2)) * (1 + k * jnp.sum(u2 ** 2))))
            return Rl / v0 * (jnp.arcsin(q) if k > 0 else jnp.arcsinh(q))
    return T


def exact_fields(name, X):
    """ Exact T (N,), grad (N, 2), hess (N, 2, 2) at points X (N, 2) away from the source """
    T = closed_form(name)
    f = jax.jit(jax.vmap(lambda x: (T(x), jax.grad(T)(x), jax.hessian(T)(x))))
    t, g, h = f(jnp.asarray(X))
    return np.asarray(t), np.asarray(g), np.asarray(h)


def errors(T, G, Hs, Te, Ge, He, R):
    out = {}
    for zone, (lo, hi) in ZONES.items():
        m = (R >= lo) & (R < hi) & np.isfinite(Te)
        if m.sum() == 0:
            continue
        L, Le = np.trace(Hs, axis1=-2, axis2=-1), np.trace(He, axis1=-2, axis2=-1)
        out[zone] = dict(
            n=int(m.sum()),
            T=float(np.abs(T - Te)[m].sum() / np.abs(Te[m]).sum()),
            G=float(np.linalg.norm(G - Ge, axis=-1)[m].sum() / np.linalg.norm(Ge, axis=-1)[m].sum()),
            L=float(np.abs(L - Le)[m].sum() / np.abs(Le[m]).sum()),
            H=float(np.linalg.norm(Hs - He, axis=(-2, -1))[m].sum() / np.linalg.norm(He, axis=(-2, -1))[m].sum()))
    return out


def fmm_derivatives(name, n, order=2):
    """ FMM on an n x n grid, derivatives by central differences (2nd order, one-sided 2nd order at the edges) """
    title, vel, xs, _, _ = P.MODELS[name]
    xs = np.asarray(xs, float)
    T, seconds = P.fmm(vel, xs, n)
    axes, X = P.grid(vel, n)
    h = axes[0][1] - axes[0][0]
    gx, gz = np.gradient(T, h, edge_order=2)
    gxx, gxz = np.gradient(gx, h, edge_order=2)
    gzx, gzz = np.gradient(gz, h, edge_order=2)
    G = np.stack([gx, gz], -1).reshape(-1, 2)
    Hs = np.stack([np.stack([gxx, 0.5 * (gxz + gzx)], -1), np.stack([0.5 * (gxz + gzx), gzz], -1)], -2).reshape(-1, 2, 2)
    Xf = X.reshape(-1, 2)
    R = np.linalg.norm(Xf - xs, axis=-1)
    ok = R > 0
    Te, Ge, He = exact_fields(name, Xf[ok])
    err = errors(T.reshape(-1)[ok], G[ok], Hs[ok], Te, Ge, He, R[ok])
    return dict(n=n, h=float(h), values=n * n, seconds=seconds, errors=err)


def nes_derivatives(prob, cfg, params):
    """ Derivatives of a trained NES by automatic differentiation, on the problem's evaluation grid """
    tt, _ = H.make_model(prob, cfg)
    e = prob.eval
    f = jax.jit(jax.vmap(lambda x, TL, gTL, HTL: (tt(params, x, TL, gTL, HTL, 0.0),
                                                   jax.grad(tt, argnums=1)(params, x, TL, gTL, HTL, 0.0),
                                                   jax.hessian(tt, argnums=1)(params, x, TL, gTL, HTL, 0.0))))
    T, G, Hs = (np.asarray(a) for a in f(e['x'], e['TL'], e['gTL'], e['HTL']))
    X = np.asarray(e['x'])
    R = np.linalg.norm(X - prob.xs, axis=-1)
    ok = R > 0
    Te, Ge, He = exact_fields(prob.name, X[ok])
    return errors(T[ok], G[ok], Hs[ok], Te, Ge, He, R[ok])
