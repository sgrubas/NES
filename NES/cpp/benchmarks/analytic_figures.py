"""The README gallery of NES-OP on analytic velocity models, redrawn with nes_cpp.

    KERAS_BACKEND=jax python NES/cpp/benchmarks/analytic_figures.py         # all 8 -> NES/cpp/docs/figures/
    KERAS_BACKEND=jax python NES/cpp/benchmarks/analytic_figures.py --only FishEye --epochs 300
    KERAS_BACKEND=jax python NES/cpp/benchmarks/analytic_figures.py --trainer cpp  # trained by nes_cpp

Same models, sources, training settings, plotting grid and style as benchmarks/analytic_models.py
(imported from there), with three differences:
  * the network is a plain MLP (improved_mlp=False), which nes_cpp can evaluate; the original figures
    use the gated MLP, so their NES isochrones come from a different network;
  * the NES traveltimes (white dashed) are computed by nes_cpp.NesModel from the exported weights;
  * the reference (black) is the closed form where it exists and nes_cpp.fmm (2nd order) elsewhere,
    instead of eikonalfm.
With --trainer keras (the default), Keras trains the network (NES_OP_<model>_cpp.png). With --trainer
cpp, nes_cpp trains it from the same start (the untrained Keras network, exported), on points drawn
the same way and with the same settings (NES_OP_<model>_cpp_trained.png); Keras then only builds the
start. Prints one JSON line per model: RMAE of nes_cpp against the reference and (if eikonalfm is
installed) the largest relative difference between the nes_cpp and eikonalfm FMM references; with
--trainer keras also the RMAE of Keras on the same network and the largest relative difference between
nes_cpp and Keras. Needs NES, matplotlib and nes_cpp.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")
import keras  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import NES  # noqa: E402
import nes_cpp  # noqa: E402
from NES.utils import Uniform_PDF  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "tools"))
from export_nes_weights import export  # noqa: E402

# The original gallery script: its CASES (title, velocity, source, closed form) and settings.
_spec = importlib.util.spec_from_file_location("gallery", REPO / "benchmarks" / "analytic_models.py")
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)


def reference(vel, xs, exact, n):
    """Plotting grid axes, receivers (n, n, dim), reference T (n, n), and the FMM table on the grid
    twice as fine (None if the closed form covers the whole domain), as gallery.reference does."""
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(vel.xmin, vel.xmax)]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    if exact == "all":
        return axes, X, vel.time(X, xs), None
    fine = [np.linspace(lo, hi, 2 * n - 1) for lo, hi in zip(vel.xmin, vel.xmax)]  # every 2nd node: plot grid
    h = [a[1] - a[0] for a in fine]
    source = tuple(int(round((c - a[0]) / dh)) for c, a, dh in zip(xs, fine, h))
    assert np.allclose([a[i] for a, i in zip(fine, source)], xs, atol=1e-9), "source is not on the FMM grid"
    v = vel(np.stack(np.meshgrid(*fine, indexing="ij"), axis=-1))
    T_fmm = nes_cpp.fmm(v, source, h, 2)
    T = T_fmm[::2, ::2]
    if exact == "lens":
        T = np.where(vel.inside(X), vel.time(X, xs), T)
    return axes, X, T, (v, source, h, T_fmm)


def build(vel, xs, args):
    """The untrained network of gallery.train, as a plain MLP."""
    keras.utils.set_random_seed(args.seed)
    nes = NES.NES_OP(xs, vel)
    nes.build_model(nl=args.nl, nu=args.nu, act="ad-gauss-1", improved_mlp=False)
    return nes


def train(vel, xs, args):
    """gallery.train with a plain MLP."""
    nes = build(vel, xs, args)
    nes.compile(lr=args.lr, decay=args.decay)
    start = time.perf_counter()
    nes.train(x_train=args.n_train, batch_size=int(np.ceil(args.n_train / args.n_batches)),
              epochs=args.epochs, verbose=0)
    return nes, time.perf_counter() - start


def train_cpp(vel, xs, args, path):
    """The same start, points (uniform in the domain, as NES draws them) and settings, trained by
    nes_cpp."""
    export(build(vel, xs, args), path)
    model = nes_cpp.NesModel(path)
    x = Uniform_PDF(vel)(args.n_train, rep=1)
    start = time.perf_counter()
    model.train(x, vel(x).reshape(-1), epochs=args.epochs, batch_size=int(np.ceil(args.n_train / args.n_batches)),
                learning_rate=args.lr, decay=args.decay, seed=args.seed)
    return model, time.perf_counter() - start


def plot(path, title, vel, xs, axes, X, T, T_ref, exact, rmae, trained_by_cpp):
    """gallery.plot, with nes_cpp named in the title and the legend."""
    x, z = axes
    fig, ax = plt.subplots(figsize=(5.9, 5.0))
    im = ax.pcolormesh(x, z, vel(X).T, cmap="viridis", shading="auto", rasterized=True)
    levels = np.linspace(np.nanmin(T_ref), np.nanmax(T_ref), 15)
    ax.contour(x, z, T_ref.T, levels, colors="black", linewidths=3.2)
    ax.contour(x, z, T.T, levels, colors="white", linewidths=1.6, linestyles="dashed")
    ax.plot(*xs, marker="*", color="red", ms=15, mec="none", clip_on=False, zorder=5)
    ax.set(xlabel="X (km)", ylabel="Z (km)", aspect="equal", xlim=(x[0], x[-1]), ylim=(z[-1], z[0]))
    by = "trained by nes_cpp" if trained_by_cpp else "nes_cpp"
    ax.set_title(f"{title} ({by}): {100 * rmae:.2f} %", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Velocity, km/s")
    reference_label = gallery.LEGEND[exact].replace("T_FMM", "T_FMM (nes_cpp)")
    ax.legend([Line2D([], [], color="black", lw=3.2), Line2D([], [], color="white", lw=1.6, ls="--")],
              [reference_label, "T_NES (nes_cpp)"], loc="upper right", facecolor="lightgrey",
              framealpha=0.85)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def eikonalfm_difference(fmm):
    """Largest relative difference between the nes_cpp FMM table and eikonalfm's (None without it)."""
    try:
        import eikonalfm
    except ImportError:
        return None
    v, source, h, T = fmm
    T_ref = eikonalfm.factored_fast_marching(v, source, h, 2) * eikonalfm.distance(v.shape, h, source,
                                                                                   indexing="ij")
    others = T > 0  # the source is 0 in nes_cpp, about 1e-16 in eikonalfm
    return float(np.max(np.abs(T[others] - T_ref[others]) / T_ref[others]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=list(gallery.CASES), help="models to run (default: all)")
    p.add_argument("--out", default=str(HERE.parent / "docs" / "figures"))
    p.add_argument("--grid", type=int, default=241, help="plotting grid nodes per axis")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--nl", type=int, default=5)
    p.add_argument("--nu", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-3)
    p.add_argument("--decay", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trainer", choices=["keras", "cpp"], default="keras", help="who trains the network")
    args = p.parse_args()
    plt.rcParams.update({"font.size": 12})
    os.makedirs(args.out, exist_ok=True)

    for name in args.only or gallery.CASES:
        title, vel, xs, exact = gallery.CASES[name]
        xs = np.asarray(xs, dtype=float)
        axes, X, T_ref, fmm = reference(vel, xs, exact, args.grid)
        keras.backend.clear_session()
        ok = np.isfinite(T_ref)
        rmae = lambda t: float(np.abs(t - T_ref)[ok].sum() / np.abs(T_ref[ok]).sum())
        result = dict(model=name)
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "model.nesw"
            if args.trainer == "cpp":
                model, seconds = train_cpp(vel, xs, args, weights)
                T = model.traveltime(X)
                result["rmae_percent"] = round(100 * rmae(T), 4)
            else:
                nes, seconds = train(vel, xs, args)
                export(nes, weights)
                T, T_keras = nes_cpp.NesModel(weights).traveltime(X), nes.Traveltime(X)
                moving = T_keras != 0  # the source: 0 in both
                result.update(rmae_percent=round(100 * rmae(T), 4), rmae_keras_percent=round(100 * rmae(T_keras), 4),
                              cpp_vs_keras=float(np.max(np.abs(T - T_keras)[moving] / np.abs(T_keras[moving]))))
        suffix = "_cpp_trained" if args.trainer == "cpp" else "_cpp"
        path = os.path.join(args.out, f"NES_OP_{name}{suffix}.png")
        plot(path, title, vel, xs, axes, X, T, T_ref, exact, rmae(T), args.trainer == "cpp")
        result.update(reference="exact" if fmm is None else gallery.REFERENCE[exact].replace("FMM", "nes_cpp FMM"),
                      fmm_vs_eikonalfm=None if fmm is None else eikonalfm_difference(fmm),
                      train_seconds=round(seconds, 1), figure=os.path.normpath(path))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
