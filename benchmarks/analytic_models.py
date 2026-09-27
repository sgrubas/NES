"""
NES-OP on the analytic velocity models of `NES.velocity`: the gallery of the README.

For every model NES-OP is trained from one source, and its isochrones (white dashed) are drawn over the reference
isochrones (black): the closed-form traveltimes where they exist, the 2nd-order factored FMM (eikonalfm) elsewhere.
The Luneburg lenses have a closed form inside the lens only, so their reference is exact inside and FMM outside.

    KERAS_BACKEND=jax python benchmarks/analytic_models.py                       # all models -> NES/data/
    KERAS_BACKEND=jax python benchmarks/analytic_models.py --only FishEye --epochs 300 --out /tmp

Prints one JSON line per model: RMAE = mean|T - T_ref| / mean|T_ref| on the plotting grid, reference, training time.
Needs matplotlib and eikonalfm.
"""
import argparse
import json
import os
import time
import numpy as np
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import NES
from NES.hpo import fmm_reference
from NES.velocity import VerticalGradient, LocAnomaly, MaxwellFishEye, LuneburgLens

BOX = dict(xmin=[-1.2, -1.2], xmax=[1.2, 1.2])

# name: (title, velocity, source, closed form: 'all' | 'lens' (inside the lens only) | None)
# Sources lie on the plotting grid (241 x 241), so that the FMM does not move them.
# LocAnomaly(vmin, vmax, ...) is vmin in the background and vmax at the centre: vmax < vmin is a low-velocity anomaly.
CASES = {
    'VerticalGradient': ('Vertical gradient', VerticalGradient(v0=1.0, a=2.0, xmin=[0., 0.], xmax=[1., 1.]),
                         [0.2, 0.2], 'all'),
    'GaussLow': ('Gaussian low-velocity anomaly', LocAnomaly(2.0, 1.0, [0., 0.], [0.35, 0.35], **BOX),
                 [-0.9, 0.0], None),
    'GaussHigh': ('Gaussian high-velocity anomaly', LocAnomaly(2.0, 3.0, [0., 0.], [0.35, 0.35], **BOX),
                  [-0.9, 0.0], None),
    'FishEye': ("Maxwell's fish-eye", MaxwellFishEye(v0=1.0, R=1.0, xmin=[-1., -1.], xmax=[1., 1.]),
                [-0.5, 0.25], 'all'),
    'HyperbolicLens': ('Hyperbolic lens', MaxwellFishEye(v0=2.0, R=1.0, high_velocity=True,
                                                         xmin=[-0.6, -0.6], xmax=[0.6, 0.6]),
                       [-0.3, 0.15], 'all'),
    'LuneburgLow': ('Luneburg lens, low velocity', LuneburgLens(v_out=2.0, R=1.0, **BOX), [0.4, -0.3], 'lens'),
    'LuneburgHigh': ('Luneburg lens, high velocity', LuneburgLens(v_out=2.0, R=1.0, n0=0.7, **BOX),
                     [0.4, -0.3], 'lens'),
    'LuneburgRim': ('Luneburg lens, source on the rim', LuneburgLens(v_out=2.0, R=1.0, **BOX), [-1.0, 0.0], 'lens'),
}
REFERENCE = {'all': 'exact', 'lens': 'exact in the lens, FMM outside', None: 'FMM'}
LEGEND = {'all': 'T_exact', 'lens': 'T_exact | T_FMM', None: 'T_FMM'}


def reference(vel, xs, exact, n):
    """ Plotting grid axes, receivers (n, n, dim) and reference traveltimes (n, n) """
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(vel.xmin, vel.xmax)]
    X = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    if exact == 'all':
        return axes, X, vel.time(X, xs)
    x_fmm, t_fmm = fmm_reference(vel, [xs], (2 * n - 1,) * vel.dim)          # every 2nd node is the plotting grid
    assert np.allclose(x_fmm[0, 0, 0, :vel.dim], xs, atol=1e-9), "source is not on the FMM grid"
    T = t_fmm[0][::2, ::2]
    if exact == 'lens':
        T = np.where(vel.inside(X), vel.time(X, xs), T)
    return axes, X, T


def train(vel, xs, args):
    keras.utils.set_random_seed(args.seed)
    nes = NES.NES_OP(xs, vel)
    nes.build_model(nl=args.nl, nu=args.nu, act='ad-gauss-1', improved_mlp=True)
    nes.compile(lr=args.lr, decay=args.decay)
    t0 = time.perf_counter()
    nes.train(x_train=args.n_train, batch_size=int(np.ceil(args.n_train / args.n_batches)), epochs=args.epochs,
              verbose=0)
    return nes, time.perf_counter() - t0


def plot(path, title, vel, xs, axes, X, T, T_ref, exact, rmae):
    x, z = axes
    fig, ax = plt.subplots(figsize=(5.9, 5.0))
    im = ax.pcolormesh(x, z, vel(X).T, cmap='viridis', shading='auto', rasterized=True)
    levels = np.linspace(np.nanmin(T_ref), np.nanmax(T_ref), 15)
    ax.contour(x, z, T_ref.T, levels, colors='black', linewidths=3.2)
    ax.contour(x, z, T.T, levels, colors='white', linewidths=1.6, linestyles='dashed')
    ax.plot(*xs, marker='*', color='red', ms=15, mec='none', clip_on=False, zorder=5)
    ax.set(xlabel='X (km)', ylabel='Z (km)', aspect='equal', xlim=(x[0], x[-1]), ylim=(z[-1], z[0]))
    ax.set_title(f'{title}: {100 * rmae:.2f} %', fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='Velocity, km/s')
    ax.legend([Line2D([], [], color='black', lw=3.2), Line2D([], [], color='white', lw=1.6, ls='--')],
              [LEGEND[exact], 'T_NES'], loc='upper right', facecolor='lightgrey', framealpha=0.85)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--only', nargs='*', choices=list(CASES), help='models to run (default: all)')
    p.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'NES', 'data'))
    p.add_argument('--grid', type=int, default=241, help='plotting grid nodes per axis')
    p.add_argument('--epochs', type=int, default=3000)
    p.add_argument('--n-train', type=int, default=4000)
    p.add_argument('--n-batches', type=int, default=8)
    p.add_argument('--nl', type=int, default=5)
    p.add_argument('--nu', type=int, default=32)
    p.add_argument('--lr', type=float, default=6e-3)
    p.add_argument('--decay', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    plt.rcParams.update({'font.size': 12})

    for name in args.only or CASES:
        title, vel, xs, exact = CASES[name]
        xs = np.asarray(xs, dtype=float)
        axes, X, T_ref = reference(vel, xs, exact, args.grid)
        keras.backend.clear_session()
        nes, seconds = train(vel, xs, args)
        T = nes.Traveltime(X)
        ok = np.isfinite(T_ref)
        rmae = float(np.abs(T - T_ref)[ok].sum() / np.abs(T_ref[ok]).sum())
        path = os.path.join(args.out, f'NES_OP_{name}.png')
        plot(path, title, vel, xs, axes, X, T, T_ref, exact, rmae)
        print(json.dumps(dict(model=name, rmae_percent=round(100 * rmae, 4), reference=REFERENCE[exact],
                              train_seconds=round(seconds, 1), backend=keras.backend.backend(),
                              figure=os.path.normpath(path))), flush=True)


if __name__ == '__main__':
    main()
