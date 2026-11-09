"""Dask: NES inference over a large chunked array of points (data parallelism).

    python NES/cpp/examples/dask_demo.py [MODEL.nesw] [--points 20000000] [--chunk 1000000]
                                         [--scheduler threads|processes|synchronous]

The receivers are a lazy dask.array of shape (points, 2) split into chunks of rows. map_blocks applies
NesModel.traveltime to every chunk, and Dask runs the chunks in parallel and streams them: the mean
and a histogram of T are computed in one pass, without keeping all traveltimes in memory.

Who runs in parallel:
  threads / processes: Dask runs one chunk per core, so OpenMP inside nes_cpp is set to one thread
                       (otherwise every chunk would start all cores again: oversubscription).
  synchronous:         Dask runs the chunks one after the other and OpenMP uses all cores in each.

A NesModel cannot be pickled, but its file path can: every process loads the model once from the
path (load_model is cached), which also works on a dask.distributed cluster. Without MODEL.nesw the
demo writes an untrained NES-OP network (4 x 50, gauss) with random weights: timing does not need a
trained model.
"""
import argparse
import functools
import os
import sys
import tempfile
import time

import numpy as np


def write_random_model(path, width=50, layers=4, seed=0):
    """A NES-OP network with random weights in the .nesw format (docs/WEIGHTS_FORMAT.md)."""
    rng = np.random.default_rng(seed)
    widths = [2] + [width] * layers
    header = ["format = nes-weights", "version = 1", "solver = NES_OP", "dim = 2", "xs = 0.5 0.1",
              "input_scale = 1.0", "factored = 1", "out_vscale = 1", "slow_min = 0.25", "slow_max = 0.5",
              "widths = " + " ".join(map(str, widths)), "act = gauss", "act_n = 1.0",
              "act_a = " + " ".join(["1.0"] * layers), "out_act = sigmoid", "out_act_n = 1.0",
              "out_act_a = 1.0"]
    arrays = []
    for n_in, n_out in zip(widths[:-1], widths[1:]):  # He-normal kernels and zero biases, as NES
        arrays += [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)), np.zeros(n_out)]
    arrays += [rng.normal(0, np.sqrt(2 / width), width), np.zeros(1)]  # head
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\nend_header\n").encode("ascii"))
        for a in arrays:
            f.write(np.ascontiguousarray(a, dtype="<f8").tobytes())


@functools.lru_cache(maxsize=None)
def load_model(path):
    """One NesModel per process, loaded on first use."""
    import nes_cpp
    return nes_cpp.NesModel(path)


def traveltime_block(points, path):
    """T for one chunk of points (rows); called by Dask for every chunk."""
    return load_model(path).traveltime(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="a .nesw file of a NES-OP model (default: random)")
    parser.add_argument("--points", type=int, default=20_000_000)
    parser.add_argument("--chunk", type=int, default=1_000_000, help="points per chunk")
    parser.add_argument("--scheduler", choices=("threads", "processes", "synchronous"), default="threads")
    args = parser.parse_args()

    if args.scheduler != "synchronous":
        os.environ["OMP_NUM_THREADS"] = "1"  # before OpenMP starts; inherited by Dask's processes
    import dask
    import dask.array as da

    model_path = args.model
    if model_path is None:
        model_path = os.path.join(tempfile.mkdtemp(), "random_op.nesw")
        write_random_model(model_path)
    if load_model(model_path).two_point:
        sys.exit("this demo draws receivers only: give a NES-OP model")

    # Lazy: nothing is computed or stored until compute() below.
    points = da.random.default_rng(0).uniform(0.0, 1.0, size=(args.points, 2), chunks=(args.chunk, 2))
    T = points.map_blocks(traveltime_block, model_path, dtype=float, drop_axis=1)
    mean = T.mean()
    histogram, edges = da.histogram(T, bins=20, range=(0.0, 0.5))

    start = time.perf_counter()
    with dask.config.set(scheduler=args.scheduler):
        mean, histogram = dask.compute(mean, histogram)  # one pass over the chunks for both
    seconds = time.perf_counter() - start
    n_chunks = points.numblocks[0]
    print(f"{args.points:.1e} points in {n_chunks} chunks, scheduler '{args.scheduler}', "
          f"{os.cpu_count()} logical CPUs: {seconds:.2f} s ({args.points / seconds:.3g} points/s)")
    print(f"mean T = {mean:.6f} s; T histogram (0 to 0.5 s, 20 bins): {histogram.tolist()}")

    # The same numbers as calling nes_cpp directly, chunk by chunk (checked on the first chunk).
    first = points.blocks[0].compute()
    assert np.array_equal(T.blocks[0].compute(scheduler="synchronous"), load_model(model_path).traveltime(first))


if __name__ == "__main__":
    main()
