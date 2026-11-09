"""Timing of NES training: nes_cpp.NesModel.train vs Keras 3 (JAX, TensorFlow, PyTorch backends, CPU).
Prints one Markdown table per training case: seconds per epoch, final loss and the speed-up of nes_cpp.

    python NES/cpp/benchmarks/bench_train.py            # full run: 20 timed epochs, median of 3 repeats
    python NES/cpp/benchmarks/bench_train.py --quick    # smoke test: 3 timed epochs, 1 repeat

    # optional extra column "cpp native": a nes_cpp built with -march=native, in the directory DIR
    NES_CPP_NATIVE_PATH=DIR python NES/cpp/benchmarks/bench_train.py

Needs nes_cpp, NES and Keras 3 with jax, tensorflow and torch installed. Keras reads KERAS_BACKEND
once, when it is imported, so every measurement runs in its own worker process
(`--backend B --dtype D` or `--cpp DIR`) and the driver (no worker arguments) collects the JSON that
each worker prints. The driver first builds the three untrained models and exports them (`--export`):
a .nesw weight file for nes_cpp and the training points with their receiver velocities.

Timing: one warm-up epoch (it compiles the Keras training step), then `train(..., epochs=E)` is
timed and divided by E. Every repeat starts from a fresh model; the time is the median over the
repeats. Keras workers rebuild the model with the same seed and precision as the export; they do not
read the exported file. The nes_cpp workers train from the exported file (always in float64) and use
their own random order of the points, so the final losses agree only roughly: they are a check that
every run trains, not a comparison of accuracy.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

BACKENDS = ("jax", "tensorflow", "torch")
DTYPES = ("float32", "float64")


def build_case(case):
    """An untrained NES model of the given case, built with seed 0 (so every process gets the same
    network and, because set_random_seed also seeds NumPy, the same training points)."""
    import keras
    import NES
    from NES.velocity import VerticalGradient
    keras.utils.set_random_seed(0)
    vel = VerticalGradient(v0=1.0, a=2.0, xmin=[0., 0.], xmax=[1., 1.])
    if case == "NES-TP 4x50":
        nes = NES.NES_TP(vel)
        nes.build_model()  # defaults: 4 x 50, reciprocity 'output'
    else:
        nes = NES.NES_OP([0.2, 0.2], vel)
        if case == "gallery NES-OP":
            nes.build_model(nl=5, nu=32, act='ad-gauss-1')  # plain MLP; the gallery also uses improved_mlp
        else:
            nes.build_model()  # defaults: 4 x 50
    return nes


# case: (points, batch size, lr, decay). The points are receivers (OP) or [source, receiver] pairs (TP).
# The last two cases use the NES defaults for lr and decay (3e-3 OP, 5e-3 TP, decay 5e-4).
CASES = {
    "gallery NES-OP": (4000, 500, 6e-3, 3e-4),   # settings of the README gallery
    "NES-OP 4x50": (20000, 2000, 3e-3, 5e-4),
    "NES-TP 4x50": (20000, 2000, 5e-3, 5e-4),
}


def training_data(nes, n):
    """Training points uniform in the velocity box and the velocity at their receivers.
    Receiver = whole row for NES-OP, second half of the row for NES-TP (rep=2). Call right after
    build_case, so that the NumPy seed is the same in every process."""
    from NES.utils import Uniform_PDF
    dim = nes.velocity.dim
    rep = 2 if type(nes).__name__ == "NES_TP" else 1
    x = Uniform_PDF(nes.velocity)(n, rep=rep)
    v = np.asarray(nes.velocity(x[:, -dim:])).reshape(-1)
    return x, v


def median_epoch_time(run_one_repeat, repeats):
    """run_one_repeat() -> (seconds per epoch, final loss). Returns the median time and the last loss."""
    times, loss = [], None
    for _ in range(repeats):
        seconds, loss = run_one_repeat()
        times.append(seconds)
    return {"s_per_epoch": statistics.median(times), "loss": loss, "all": times}


def export_models(out_dir):
    """Writes <case>.nesw and <case>.npz (x, v) for every case into out_dir."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from export_nes_weights import export
    for case, (n, *_rest) in CASES.items():
        nes = build_case(case)
        x, v = training_data(nes, n)
        stem = Path(out_dir) / case.replace(" ", "_")
        export(nes, stem.with_suffix(".nesw"))
        np.savez(stem.with_suffix(".npz"), x=x, v=v)


def keras_worker(dtype, epochs, repeats):
    """Times nes.train for every case with the backend in KERAS_BACKEND."""
    import keras
    if dtype == "float64":  # as in the float64 fixture of tests/python/test_nes_model.py
        keras.config.set_floatx("float64")
        keras.config.set_dtype_policy("float64")
        if keras.backend.backend() == "jax":
            import jax
            jax.config.update("jax_enable_x64", True)
    results = {}
    for case, (n, batch, lr, decay) in CASES.items():
        def one_repeat():
            nes = build_case(case)
            x, _ = training_data(nes, n)
            nes.compile(lr=lr, decay=decay)
            nes.train(x, batch_size=batch, epochs=1, verbose=0)  # warm-up: compiles the training step
            start = time.perf_counter()
            history = nes.train(x, batch_size=batch, epochs=epochs, verbose=0)
            return (time.perf_counter() - start) / epochs, float(history.history["loss"][-1])
        try:
            results[case] = median_epoch_time(one_repeat, repeats)
        except Exception as error:  # e.g. an op a backend lacks in float64
            print(f"{case}: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr)
            results[case] = None
    return results


def cpp_worker(model_dir, epochs, repeats):
    """Times nes_cpp.NesModel.train; the number of threads comes from OMP_NUM_THREADS (set by the driver)."""
    import nes_cpp
    results = {}
    for case, (n, batch, lr, decay) in CASES.items():
        stem = Path(model_dir) / case.replace(" ", "_")
        data = np.load(stem.with_suffix(".npz"))
        x, v = data["x"], data["v"]

        def one_repeat():
            model = nes_cpp.NesModel(stem.with_suffix(".nesw"))  # fresh weights
            model.train(x, v, epochs=1, batch_size=batch, learning_rate=lr, decay=decay, seed=0)  # warm-up
            start = time.perf_counter()
            history = model.train(x, v, epochs=epochs, batch_size=batch, learning_rate=lr, decay=decay, seed=1)
            return (time.perf_counter() - start) / epochs, float(history[-1])
        try:
            results[case] = median_epoch_time(one_repeat, repeats)
        except Exception as error:
            print(f"{case}: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr)
            results[case] = None
    return results


def launch(args, env_extra):
    """Runs this script as a worker process and returns the JSON it prints on its last line.
    The worker's stderr (warnings, failures) passes through to ours."""
    env = dict(os.environ, **env_extra)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    out = subprocess.run([sys.executable, __file__] + args, env=env, check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def fmt_time(r):
    return "failed" if r is None else f"{r['s_per_epoch']:.3g}"


def fmt_loss(r):
    return "failed" if r is None else f"{r['loss']:.2e}"


def fmt_speedup(r, ref):
    """Time of the row's run over the time of the nes_cpp column `ref` (None if either failed)."""
    if r is None or ref is None:
        return "-"
    return f"{r['s_per_epoch'] / ref['s_per_epoch']:.1f}"


def print_tables(results):
    """One table per case. results: {column name: {case: {'s_per_epoch', 'loss'} or None}}.
    The speed-up columns are (time of the row) / (time of the nes_cpp column)."""
    cpp_cols = [k for k in results if k.startswith("cpp")]
    for case in CASES:
        n, batch, lr, decay = CASES[case]
        print(f"\n### {case}: {n} points, batch {batch}, lr {lr:g}, decay {decay:g}\n")
        print("| run | s/epoch | final loss | " + " | ".join(f"speed-up of {c}" for c in cpp_cols) + " |")
        print("|" + "---|" * (3 + len(cpp_cols)))
        for name, res in results.items():
            r = res[case]
            ups = " | ".join(fmt_speedup(r, results[c][case]) if name != c else "1.0" for c in cpp_cols)
            print(f"| {name} | {fmt_time(r)} | {fmt_loss(r)} | {ups} |")


def versions():
    import importlib.metadata as md
    cpu = platform.processor()
    if os.path.exists("/proc/cpuinfo"):  # Linux: the model name, not just 'x86_64'
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    found = ", ".join(f"{n} {md.version(n)}" for n in
                      ("numpy", "keras", "jax", "tensorflow-cpu", "torch", "nes-cpp"))
    return f"{cpu}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}, {found}"


def driver(epochs, repeats, quick):
    print(versions())
    print(f"\nSeconds per training epoch (median of {repeats}, {epochs} timed epochs after 1 warm-up epoch). "
          "cpp all = all threads, cpp 1 = OMP_NUM_THREADS=1. Speed-up = row time / nes_cpp time. "
          "Keras: NES defaults (jit_compile auto), verbose=0.")
    flag = ["--epochs", str(epochs), "--repeats", str(repeats)]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        # The export only reads weights, so any backend will do; JAX is the fastest to start.
        subprocess.run([sys.executable, __file__, "--export", tmp], check=True,
                       env=dict(os.environ, KERAS_BACKEND="jax", TF_CPP_MIN_LOG_LEVEL="2"))
        results["cpp all"] = launch(["--cpp", tmp] + flag, {})
        results["cpp 1"] = launch(["--cpp", tmp] + flag, {"OMP_NUM_THREADS": "1"})
        native = os.environ.get("NES_CPP_NATIVE_PATH")
        if native:  # an -march=native build; put it first on the module search path
            path = native + os.pathsep + os.environ.get("PYTHONPATH", "")
            results["cpp native"] = launch(["--cpp", tmp] + flag, {"PYTHONPATH": path})
        for backend in BACKENDS:
            for dtype in DTYPES:
                print(f"running {backend} {dtype} ...", file=sys.stderr)
                results[f"{backend} {dtype}"] = launch(["--backend", backend, "--dtype", dtype] + flag,
                                                       {"KERAS_BACKEND": backend})
    print_tables(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="3 timed epochs, 1 repeat")
    parser.add_argument("--epochs", type=int, help="timed epochs (default 20, --quick 3)")
    parser.add_argument("--repeats", type=int, help="repeats (default 3, --quick 1)")
    parser.add_argument("--backend", choices=BACKENDS, help="worker: time Keras (KERAS_BACKEND is set by the driver)")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--cpp", metavar="DIR", help="worker: time nes_cpp on the models exported to DIR")
    parser.add_argument("--export", metavar="DIR", help="worker: export the models to DIR")
    args = parser.parse_args()
    epochs = args.epochs or (3 if args.quick else 20)
    repeats = args.repeats or (1 if args.quick else 3)
    if args.export:
        export_models(args.export)
    elif args.cpp:
        print(json.dumps(cpp_worker(args.cpp, epochs, repeats)))
    elif args.backend:
        print(json.dumps(keras_worker(args.dtype, epochs, repeats)))
    else:
        driver(epochs, repeats, args.quick)


if __name__ == "__main__":
    main()
