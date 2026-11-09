"""Timing of NES inference: nes_cpp.NesModel vs Keras 3 (JAX, TensorFlow, PyTorch backends, CPU).
Prints Markdown tables: time per call and the speed-up of nes_cpp over each Keras backend.

    python NES/cpp/benchmarks/bench_nes.py            # full run (long: up to N = 1e7)
    python NES/cpp/benchmarks/bench_nes.py --quick    # smoke test: N = 1e3, 1e4 and 2 repeats

    # optional extra column "cpp native": a nes_cpp built with -march=native, in the directory DIR
    NES_CPP_NATIVE_PATH=DIR python NES/cpp/benchmarks/bench_nes.py

Needs nes_cpp, NES and Keras 3 with jax, tensorflow and torch installed. Keras reads KERAS_BACKEND
once, when it is imported, so every measurement runs in its own worker process
(`--backend B --dtype D` or `--cpp DIR`) and the driver (no worker arguments) collects the JSON that
each worker prints. Models: an untrained NES-OP with the NES defaults and the saved NES-TP model
tests/Models/Marmousi. The driver exports both to .nesw files once (`--export`), for nes_cpp.

Quantities: "T" is the traveltime alone; "T + gradient" is the traveltime and its gradient (OP: 'G';
TP: 'Gr' and 'Gs'). Keras computes them in one nes.predict call. nes_cpp has two methods, so it
calls them one after the other (its gradient computes T again internally). Keras runs with the batch
size NES uses by default (100000 points per batch).

tests/Models/Marmousi is a local model, not in git; the script stops if it is missing.
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

REPO = Path(__file__).resolve().parents[3]
TP_MODEL = REPO / "tests" / "Models" / "Marmousi"
SIZES = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
TIME_LIMIT = 120.0     # s; after one run slower than this, larger N are skipped
MODELS = ("OP", "TP")
QUANTITIES = ("T", "T + gradient")
BACKENDS = ("jax", "tensorflow", "torch")
DTYPES = ("float32", "float64")


def sizes_for(quick):
    return SIZES[:2] if quick else SIZES


def repeats_for(n, quick):
    """Median over 5 runs for N <= 1e5, 3 for 1e6, 1 for 1e7 (2 for every N in --quick)."""
    if quick:
        return 2
    return 5 if n <= 10 ** 5 else 3 if n <= 10 ** 6 else 1


def make_points(model, box, n):
    """OP: receivers uniform in the box. TP: [source, receiver] pairs, both uniform in the box."""
    lo, hi = np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)
    if model == "TP":
        lo, hi = np.tile(lo, 2), np.tile(hi, 2)
    return np.random.default_rng(0).uniform(lo, hi, size=(n, len(lo)))  # same points in every process


def median_time(run, repeats):
    """One untimed warm-up call (JIT compilation), then the median of `repeats` timed calls.
    Returns (median, slowest) in seconds."""
    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times), max(times)


def sweep(model, quantity, box, run, quick):
    """{N: seconds or None} for one configuration; run(x) is the call to time.
    An error marks the cell None and ends the sweep; so does a run slower than TIME_LIMIT."""
    results = {}
    for n in sizes_for(quick):
        x = make_points(model, box, n)
        try:
            seconds, slowest = median_time(lambda: run(x), repeats_for(n, quick))
        except Exception as error:  # e.g. an op a backend lacks in float64, or out of memory
            print(f"{model} {quantity} N={n}: {type(error).__name__}: {str(error)[:200]}", file=sys.stderr)
            results[n] = None
            break
        results[n] = seconds
        if slowest > TIME_LIMIT:
            print(f"{model} {quantity}: a run took {slowest:.0f} s, larger N skipped", file=sys.stderr)
            break
    return results


def build_keras_models():
    """NES-OP (untrained, NES defaults) and NES-TP (saved model), in the current precision."""
    import keras
    import NES
    from NES.velocity import VerticalGradient
    if not TP_MODEL.is_dir():
        sys.exit(f"{TP_MODEL} not found: this local NES-TP model is needed (it is not in git)")
    keras.utils.set_random_seed(0)
    op = NES.NES_OP(xs=[0.3, 0.2], velocity=VerticalGradient(v0=2.0, a=1.0, xmin=[0., 0.], xmax=[1., 1.]))
    op.build_model()  # 4 hidden layers x 50, ad-gauss-1
    return {"OP": op, "TP": NES.NES_TP.load(str(TP_MODEL))}


def export_models(out_dir):
    """Writes OP.nesw, TP.nesw and boxes.json (the velocity boxes, for the points) into out_dir."""
    sys.path.insert(0, str(REPO / "NES" / "cpp" / "tools"))
    from export_nes_weights import export
    boxes = {}
    for name, nes in build_keras_models().items():
        export(nes, Path(out_dir) / f"{name}.nesw")
        boxes[name] = [[float(v) for v in nes.velocity.xmin], [float(v) for v in nes.velocity.xmax]]
    (Path(out_dir) / "boxes.json").write_text(json.dumps(boxes))


def keras_worker(dtype, quick):
    """Times nes.Traveltime and nes.predict(...) for both models with the backend in KERAS_BACKEND."""
    import keras
    if dtype == "float64":  # as in the float64 fixture of tests/python/test_nes_model.py
        keras.config.set_floatx("float64")
        keras.config.set_dtype_policy("float64")
        if keras.backend.backend() == "jax":
            import jax
            jax.config.update("jax_enable_x64", True)
    outputs = {"OP": ("T", "G"), "TP": ("T", "Gr", "Gs")}
    results = {}
    for name, nes in build_keras_models().items():
        box = [nes.velocity.xmin, nes.velocity.xmax]
        results[f"{name}|T"] = sweep(
            name, "T", box, lambda x: nes.Traveltime(x), quick)
        results[f"{name}|T + gradient"] = sweep(
            name, "T + gradient", box,
            lambda x: nes.predict(x, outputs[name]), quick)
    return results


def cpp_worker(model_dir, quick):
    """Times nes_cpp.NesModel; the number of threads comes from OMP_NUM_THREADS (set by the driver)."""
    import nes_cpp
    boxes = json.loads((Path(model_dir) / "boxes.json").read_text())
    results = {}
    for name in MODELS:
        model = nes_cpp.NesModel(Path(model_dir) / f"{name}.nesw")

        def traveltime_and_gradient(x):
            model.traveltime(x)
            model.gradient(x)  # for TP this is the pair (gradient_r, gradient_s) in one call

        results[f"{name}|T"] = sweep(name, "T", boxes[name], model.traveltime, quick)
        results[f"{name}|T + gradient"] = sweep(name, "T + gradient", boxes[name],
                                                traveltime_and_gradient, quick)
    return results


def launch(args, env_extra):
    """Runs this script as a worker process and returns the JSON it prints on its last line.
    The worker's stderr (warnings, skipped sizes) passes through to ours."""
    env = dict(os.environ, **env_extra)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    out = subprocess.run([sys.executable, __file__] + args, env=env, check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def fmt(seconds):
    return seconds if isinstance(seconds, str) else "failed" if seconds is None else f"{seconds:.3g}"


def print_tables(results, quick):
    """Two tables per model and quantity: seconds per call, then the speed-up of nes_cpp over each Keras
    run. results: {column name: {'model|quantity': {N: seconds}}}; JSON turned the N keys into strings."""
    keras_names = [k for k in results if not k.startswith("cpp")]
    for model in MODELS:
        for quantity in QUANTITIES:
            key = f"{model}|{quantity}"
            times = {(k, n): results[k][key].get(str(n), "skipped") for k in results for n in sizes_for(quick)}
            print(f"\n### NES-{model}, {quantity}: seconds per call\n")
            print("| N | " + " | ".join(results) + " |")
            print("|" + "---|" * (1 + len(results)))
            for n in sizes_for(quick):
                print(f"| {n:.0e} | " + " | ".join(fmt(times[k, n]) for k in results) + " |")
            print(f"\n### NES-{model}, {quantity}: speed-up of nes_cpp (Keras time / nes_cpp time)\n")
            print("| N | nes_cpp | " + " | ".join(keras_names) + " |")
            print("|" + "---|" * (2 + len(keras_names)))
            for n in sizes_for(quick):
                for cpp in ("cpp all", "cpp 1"):
                    row = []
                    for k in keras_names:
                        t_cpp, t_k = times[cpp, n], times[k, n]
                        ok = isinstance(t_cpp, float) and isinstance(t_k, float)
                        row.append(f"{t_k / t_cpp:.1f}" if ok else "-")
                    print(f"| {n:.0e} | {cpp} | " + " | ".join(row) + " |")


def versions():
    import importlib.metadata as md
    cpu = platform.processor()
    if os.path.exists("/proc/cpuinfo"):  # Linux: the model name, not just 'x86_64'
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    found = ", ".join(f"{n} {md.version(n)}" for n in
                      ("numpy", "keras", "jax", "tensorflow-cpu", "torch", "nes-cpp"))
    return f"{cpu}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}, {found}"


def driver(quick):
    print(versions())
    print(f"\nSeconds per call (median). skipped: a smaller N took more than {TIME_LIMIT:.0f} s; "
          "failed: an error (message on stderr). Keras: NES default batch size (100000); "
          "cpp all = all threads, cpp 1 = OMP_NUM_THREADS=1.")
    flag = ["--quick"] if quick else []
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        # The export only reads weights, so any backend will do; JAX is the fastest to start.
        subprocess.run([sys.executable, __file__, "--export", tmp], check=True,
                       env=dict(os.environ, KERAS_BACKEND="jax"))
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
    print_tables(results, quick)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="N = 1e3, 1e4 and 2 repeats")
    parser.add_argument("--backend", choices=BACKENDS, help="worker: time Keras (KERAS_BACKEND is set by the driver)")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--cpp", metavar="DIR", help="worker: time nes_cpp on the models exported to DIR")
    parser.add_argument("--export", metavar="DIR", help="worker: export the models to DIR")
    args = parser.parse_args()
    if args.export:
        export_models(args.export)
    elif args.cpp:
        print(json.dumps(cpp_worker(args.cpp, args.quick)))
    elif args.backend:
        print(json.dumps(keras_worker(args.dtype, args.quick)))
    else:
        driver(args.quick)


if __name__ == "__main__":
    main()
