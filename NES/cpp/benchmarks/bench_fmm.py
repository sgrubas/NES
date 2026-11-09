"""Timing of the factored FMM: eikonalfm vs nes_cpp.fmm (one source, one thread) vs nes_cpp.fmm_multi
(many sources, OpenMP). Prints Markdown tables with medians of repeated runs.

    python NES/cpp/benchmarks/bench_fmm.py

Needs nes_cpp, eikonalfm and numpy. The velocity is smooth and heterogeneous: 2 km/s plus a vertical
gradient of 1/s and a lateral ripple. eikonalfm is timed for its solver call only (it returns tau1;
NES multiplies by eikonalfm.distance afterwards), while nes_cpp returns T.
"""
import os
import platform
import statistics
import subprocess
import sys
import time

import eikonalfm
import numpy as np

import nes_cpp

H = 0.01  # km


def velocity(shape):
    axes = [np.arange(n) * H for n in shape]
    x = np.meshgrid(*axes, indexing="ij")
    return 2.0 + x[-1] + 0.3 * np.sin(10.0 * x[0])


def median_seconds(run, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def single_source():
    print("| Grid | Order | eikonalfm (s) | nes_cpp.fmm (s) | eikonalfm / nes_cpp |")
    print("|---|---|---|---|---|")
    for shape, repeats in [((201, 201), 21), ((501, 501), 7), ((1001, 1001), 3), ((101, 101, 101), 3)]:
        v = velocity(shape)
        source = (shape[0] // 2,) + (0,) * (len(shape) - 1)  # top centre
        h = (H,) * len(shape)
        for order in (1, 2):
            t_ref = median_seconds(lambda: eikonalfm.factored_fast_marching(v, source, h, order), repeats)
            t_cpp = median_seconds(lambda: nes_cpp.fmm(v, source, h, order), repeats)
            grid = " x ".join(map(str, shape))
            print(f"| {grid} | {order} | {t_ref:.3f} | {t_cpp:.3f} | {t_ref / t_cpp:.2f} |")


# Times fmm_multi in a fresh process, so that OMP_NUM_THREADS is read when OpenMP starts.
TIME_FMM_MULTI = """
import statistics, sys, time
import numpy as np
import nes_cpp
v, sources = np.load(sys.argv[1]), np.load(sys.argv[2])
times = []
for _ in range(int(sys.argv[3])):
    start = time.perf_counter()
    nes_cpp.fmm_multi(v, sources, (0.01, 0.01), 2)
    times.append(time.perf_counter() - start)
print(statistics.median(times))
"""


def many_sources(tmp_dir, shape=(501, 501), n_sources=32, repeats=3):
    v = velocity(shape)
    sources = np.stack([np.linspace(0, shape[0] - 1, n_sources).round().astype(np.int64),
                        np.zeros(n_sources, dtype=np.int64)], axis=1)  # along the top
    h = (H, H)
    t_ref = median_seconds(lambda: [eikonalfm.factored_fast_marching(v, tuple(s), h, 2) for s in sources],
                           repeats)
    t_loop = median_seconds(lambda: [nes_cpp.fmm(v, s, h, 2) for s in sources], repeats)

    np.save(os.path.join(tmp_dir, "v.npy"), v)
    np.save(os.path.join(tmp_dir, "sources.npy"), sources)
    grid = " x ".join(map(str, shape))
    print(f"\n{n_sources} sources on {grid}, order 2:\n")
    print("| Method | Threads | Time (s) | Speedup vs eikonalfm |")
    print("|---|---|---|---|")
    print(f"| eikonalfm, Python loop | 1 | {t_ref:.3f} | 1.00 |")
    print(f"| nes_cpp.fmm, Python loop | 1 | {t_loop:.3f} | {t_ref / t_loop:.2f} |")
    for threads in (1, 2, 4, 8):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        out = subprocess.run([sys.executable, "-c", TIME_FMM_MULTI, os.path.join(tmp_dir, "v.npy"),
                              os.path.join(tmp_dir, "sources.npy"), str(repeats)],
                             env=env, check=True, capture_output=True, text=True).stdout
        t = float(out)
        print(f"| nes_cpp.fmm_multi | {threads} | {t:.3f} | {t_ref / t:.2f} |")


def main():
    import tempfile

    print(f"Python {platform.python_version()}, {platform.system()} {platform.release()}, "
          f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
          f"nes_cpp {nes_cpp.__version__}, NumPy {np.__version__}\n")
    single_source()
    with tempfile.TemporaryDirectory() as tmp_dir:
        many_sources(tmp_dir)


if __name__ == "__main__":
    main()
