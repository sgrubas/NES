"""Plain Python threads calling nes_cpp: the kernels release the GIL, so the threads run in parallel.

    python NES/cpp/examples/threads_demo.py [--sources 32] [--size 501]

Each thread takes the next source from a ThreadPoolExecutor and solves it with nes_cpp.fmm (one FMM
solve is sequential, so threads are the only parallelism here). For contrast, the same is timed with
eikonalfm if it is installed: it is C++ too, but it keeps the GIL during its solve, so its threads
take turns and do not speed up. At the end the threaded results are checked bit for bit against
nes_cpp.fmm_multi, which runs the same solves with OpenMP inside C++.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import nes_cpp

H = (0.01, 0.01)  # km


def velocity(n):
    """n x n grid, 2 km/s plus a vertical gradient and a lateral ripple (as benchmarks/bench_fmm.py)."""
    x, z = np.meshgrid(np.arange(n) * H[0], np.arange(n) * H[1], indexing="ij")
    return 2.0 + z + 0.3 * np.sin(10.0 * x)


def solve_with_threads(solve, sources, n_threads):
    """solve(source) for every source on `n_threads` Python threads; returns (results, seconds)."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        results = list(pool.map(solve, sources))  # in the order of `sources`
    return results, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=int, default=32)
    parser.add_argument("--size", type=int, default=501, help="grid nodes per axis")
    args = parser.parse_args()

    v = velocity(args.size)
    sources = [(int(i), 0) for i in np.linspace(0, args.size - 1, args.sources)]  # along the top
    solvers = {"nes_cpp.fmm": lambda s: nes_cpp.fmm(v, s, H, 2)}
    try:
        import eikonalfm
        solvers["eikonalfm"] = lambda s: eikonalfm.factored_fast_marching(v, s, H, 2)
    except ImportError:
        print("eikonalfm is not installed: timing nes_cpp only")

    print(f"{args.sources} FMM solves (order 2) on {args.size} x {args.size}, {os.cpu_count()} logical CPUs\n")
    print("| Solver | Threads | Time (s) | Speed-up vs 1 thread |")
    print("|---|---|---|---|")
    same = None
    for name, solve in solvers.items():
        solve(sources[0])  # warm-up: first call loads libraries
        one_thread = None
        for n_threads in (1, 2, 4, 8):
            results, seconds = solve_with_threads(solve, sources, n_threads)
            one_thread = one_thread or seconds
            print(f"| {name} | {n_threads} | {seconds:.2f} | {one_thread / seconds:.2f} |")
        if name == "nes_cpp.fmm":
            table = nes_cpp.fmm_multi(v, np.array(sources), H, 2)
            same = all(np.array_equal(T, table[k]) for k, T in enumerate(results))
    print(f"\nThreaded nes_cpp.fmm equals nes_cpp.fmm_multi bit for bit: {same}")


if __name__ == "__main__":
    main()
