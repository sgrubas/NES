"""Ray: multi-source FMM traveltime tables as tasks (task parallelism).

    python NES/cpp/examples/ray_demo.py [--sources 256] [--batch 16]

The smoothed Marmousi model (NES/data, 1361 x 281 nodes, 12.5 m) is put once into Ray's object store
with ray.put. Every task receives a batch of sources and computes their traveltime table with
nes_cpp.fmm_multi on one core (OMP_NUM_THREADS=1 in the task's environment); Ray schedules the tasks
over the CPUs, or over the nodes of a cluster with the same code. The velocity reaches the tasks as a
read-only view of the shared object-store memory, not as one copy per task.

For comparison, the same table is computed with one nes_cpp.fmm_multi call in this process (OpenMP
over the sources), and the two results are checked bit for bit.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import ray

import nes_cpp

MARMOUSI = Path(__file__).resolve().parents[2] / "data" / "Marmousi_Pwave_smooth_12_5m.npy"
H = (0.0125, 0.0125)  # km


@ray.remote(num_cpus=1, runtime_env={"env_vars": {"OMP_NUM_THREADS": "1"}})
def tables(v, sources):
    """Traveltime tables (len(sources), n_x, n_z) for a batch of sources, on one core."""
    return nes_cpp.fmm_multi(v, sources, H, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16, help="sources per task")
    args = parser.parse_args()

    v = np.load(MARMOUSI) / 1000.0  # m/s -> km/s, shape (n_x, n_z)
    sources = np.stack([np.linspace(0, v.shape[0] - 1, args.sources).round(),  # along the surface
                        np.zeros(args.sources)], axis=1).astype(np.int64)
    batches = [sources[i:i + args.batch] for i in range(0, len(sources), args.batch)]

    ray.init(num_cpus=os.cpu_count())
    v_ref = ray.put(v)  # stored once; every task gets this reference instead of a copy of v
    ray.get(tables.remote(v_ref, sources[:1]))  # warm-up: start the worker processes

    start = time.perf_counter()
    table = np.concatenate(ray.get([tables.remote(v_ref, batch) for batch in batches]))
    seconds_ray = time.perf_counter() - start

    start = time.perf_counter()
    reference = nes_cpp.fmm_multi(v, sources, H, 2)
    seconds_openmp = time.perf_counter() - start

    print(f"{args.sources} sources on Marmousi {v.shape[0]} x {v.shape[1]}, {os.cpu_count()} logical CPUs")
    print(f"Ray, {len(batches)} tasks of {args.batch} sources, one core each: {seconds_ray:.2f} s")
    print(f"nes_cpp.fmm_multi in one process, OpenMP:         {seconds_openmp:.2f} s")
    print(f"tables equal bit for bit: {np.array_equal(table, reference)}; "
          f"table size {table.nbytes / 1e6:.0f} MB")
    ray.shutdown()


if __name__ == "__main__":
    main()
