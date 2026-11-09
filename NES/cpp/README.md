# nes-cpp: C++ eikonal kernels for NES

C++ implementations of two eikonal solvers, with Python bindings:

- **Factored fast marching (FMM), orders 1 and 2**, 2D and 3D (Treister & Haber 2016), validated against [eikonalfm](https://github.com/kevinganster/eikonalfm).
- **NES inference**: traveltime T and its gradient ∇T from a trained Neural Eikonal Solver (`NES_OP`, `NES_TP`), using weights exported from Python NES.
- **NES training**: a minimal training loop for the same networks (the NES loss, its gradient by backpropagation, Adam), validated against Keras 3 autodiff. The trained weights go back into Python NES.

The aim is speed, measured honestly against eikonalfm and Keras. Multi-source FMM tables run in parallel with OpenMP; Dask and Ray examples show scale-out.

> **Status:** factored FMM of orders 1 and 2 in 2-D and 3-D, one source or many sources in parallel, in C++ and from Python (`nes_cpp.fmm`, `nes_cpp.fmm_multi`; see [`docs/FMM.md`](docs/FMM.md)). NES inference computes T and ∇T (reverse mode) from exported NES-OP and NES-TP models ([`docs/NES_INFERENCE.md`](docs/NES_INFERENCE.md)). NES training runs in C++ from an exported untrained (or trained) network ([`docs/NES_TRAINING.md`](docs/NES_TRAINING.md)). A CI workflow builds and tests on Linux, with sanitizers and a formatting check. The `examples/` show the kernels under Python threads, Dask and Ray, and the NES-OP gallery of the main README is redrawn with nes_cpp twice: networks trained by Keras and evaluated by nes_cpp, and networks trained by nes_cpp (`benchmarks/analytic_figures.py`).

## Layout
| Path | Contents |
|---|---|
| `docs/FMM.md` | factored discretization, 1st- and 2nd-order updates, fallbacks, heap |
| `docs/NES_INFERENCE.md` | network equations as in `NES/layers.py`, reverse-mode ∇T |
| `docs/NES_TRAINING.md` | NES loss, its gradient (double backpropagation), Adam, reproducibility |
| `docs/WEIGHTS_FORMAT.md` | exported weight file format (versioned) |
| `include/nes_cpp/`, `src/` | C++ library: `fmm.*`, `nes_model.*` (inference), `nes_train.cpp` (training); internal headers in `src/` |
| `bindings/module.cpp` | pybind11 module `nes_cpp` |
| `tools/export_nes_weights.py` | NES model (saved folder or in memory) → weight file |
| `tools/import_nes_weights.py` | weight file → the weights of an NES model in memory |
| `tests/cpp/` | Catch2 tests (analytic FMM, loader, activations, loss gradient, Adam) |
| `tests/python/` | pytest: parity with eikonalfm and NES (T, ∇T, loss gradient), convergence, training |
| `examples/` | threads, Dask and Ray demos (see "Scale-out") |
| `benchmarks/` | FMM, NES inference and NES training timing scripts; `analytic_figures.py` redraws the NES gallery with nes_cpp |

## Conventions
- Points: `(..., dim)`, coordinates `(x, z)` or `(x, y, z)`, z last (as in `NES.velocity`).
- Grids: arrays of shape `(n_x, [n_y,] n_z)`, C-order, spacing `h` per axis; the FMM source is a grid index.
- Everything is float64 in C++.

## Build and test
**C++ tests (CLion, Windows):** open `NES/cpp` as a CMake project. Toolchain: Qt MinGW 13.1 or Visual Studio. Catch2 and Eigen are fetched on first configure. Run the `tests` target.

**Python module** (on Windows, python.org CPython is built with MSVC, so build the module with MSVC; `pip install` picks it automatically when Visual Studio C++ tools are installed):
From the repo root:
```
pip install -e ".[jax]" eikonalfm==0.9.8   # references for the parity tests: NES itself and eikonalfm
pip install ./NES/cpp[test]
pytest NES/cpp/tests/python
```
For a faster build tuned to the CPU it runs on (AVX2, FMA, AVX-512 where available; about 3 times faster NES inference and 1.8 to 2.9 times faster NES training, see "Benchmarks"):
```
pip install ./NES/cpp --config-settings=cmake.define.CMAKE_CXX_FLAGS=-march=native
```
That binary may not run on other CPUs, and its results differ from the portable build in the last bits (fused multiply-add); the parity tests pass with both.

Tests that need eikonalfm or NES are skipped when those packages are missing. eikonalfm 0.9.8 is used because 0.9.5, pinned in the NES notebooks, has no wheels for Python 3.10+; its solver code is the same (only C++ style changes).

On Windows the module uses MSVC's OpenMP runtime (`vcomp140.dll`), which comes with the Microsoft Visual C++ Redistributable.

**CI** (`.github/workflows/cpp.yml`, Linux, runs when `NES/cpp/**` or the workflow changes):
- *Build and test:* C++ tests in a Release build with warnings as errors; then `pip install` of NES, `jax[cpu]`, eikonalfm and nes-cpp, and all Python tests. The step fails if eikonalfm or NES cannot be imported, so the parity tests cannot be skipped.
- *Sanitizers:* C++ tests in a Debug build with AddressSanitizer (memory errors, leaks) and UndefinedBehaviorSanitizer; any report fails the job.
- *clang-format:* formatting check with clang-format 23.1.3 and `NES/cpp/.clang-format`.

To reproduce the last two locally (Linux or WSL; MinGW has no AddressSanitizer), from the repo root:
```
cmake -S NES/cpp -B NES/cpp/build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build NES/cpp/build-asan && ctest --test-dir NES/cpp/build-asan --output-on-failure
pip install clang-format==23.1.3
clang-format -i $(git ls-files 'NES/cpp/*.cpp' 'NES/cpp/*.hpp')   # rewrites files in place
```
CLion's "Reformat Code" reads `.clang-format` but uses its own bundled clang-format, whose version may differ from CI's; run the pip-installed `clang-format -i` above before committing.

**Usage:**
```python
import numpy as np, nes_cpp
T = nes_cpp.fmm(v, source=(10, 0), h=(0.0125, 0.0125), order=2)  # v: (n_x, n_z), T: same shape
table = nes_cpp.fmm_multi(v, sources=np.array([[10, 0], [50, 0]]), h=(0.0125, 0.0125), order=2)
# table: (2, n_x, n_z); threads: OMP_NUM_THREADS (default: all logical cores)

model = nes_cpp.NesModel("marmousi_op.nesw")   # from tools/export_nes_weights.py
T = model.traveltime(xr)                         # xr: (..., 2) for NES-OP, (..., 4) [xs, xr] for NES-TP
G = model.gradient(xr)                           # dT/dxr (NES-OP) or [dT/dxs, dT/dxr] (NES-TP)
Gr, Gs = model.gradient_r(xr), model.gradient_s(xr)  # NES-TP only
```

**Training** (see [`docs/NES_TRAINING.md`](docs/NES_TRAINING.md)): export an NES model built in Python (untrained, or trained, to continue), train it in C++, and read the result back into NES:
```python
from NES.utils import Uniform_PDF
from export_nes_weights import export            # NES/cpp/tools
from import_nes_weights import load_into
nes = NES.NES_OP(xs, vel); nes.build_model(nl=5, nu=32)
export(nes, "start.nesw")
model = nes_cpp.NesModel("start.nesw")
x = Uniform_PDF(vel)(4000, rep=1)                 # training points: receivers (NES-OP) or [xs, xr] (NES-TP)
history = model.train(x, vel(x[:, -2:]).ravel(), epochs=3000, batch_size=500, learning_rate=6e-3, decay=3e-4)
model.save("trained.nesw")                        # the same file format
load_into(nes, "trained.nesw")                    # nes now has the trained weights
```
`train` takes the velocity at the receiver of every point, so nes_cpp needs no velocity model. Options: `epochs`, `batch_size` (NES/Keras default 32), `learning_rate` (NES default: 3e-3 NES-OP, 5e-3 NES-TP), `decay` (inverse-time, 5e-4), `beta_1`, `beta_2`, `epsilon` (Adam), `seed` (order of the points). It returns the mean loss of every epoch. The same seed and data give the same weights, bit for bit, with any number of threads. For other optimizers, `loss_and_gradient(x, v)` returns the loss and its gradient, and `parameters()` and `set_parameters(p)` read and write the weights, all as flat arrays in the order given in `docs/NES_TRAINING.md`.

**Exporting an NES model** (Python with NES installed; see [`docs/WEIGHTS_FORMAT.md`](docs/WEIGHTS_FORMAT.md)):
```
python NES/cpp/tools/export_nes_weights.py path/to/saved_model marmousi_op.nesw
```
or `export(nes, "model.nesw")` from `tools/export_nes_weights.py` for a model in memory. Folders saved by Keras-3 NES and by the original TensorFlow/Keras-2 NES both work. Supported: `TraveltimeNet` with reciprocity `None` or `'output'`, without `improved_mlp`, activations `gauss`, `tanh`, `sigmoid`, `sin`, `swish`, `linear`, `softplus`, `elu`, `relu`; anything else raises `ValueError` at export.

**Python ↔ C++ boundary** (`bindings/module.cpp`):
- A C-contiguous float64 velocity array is passed to C++ without a copy. Other arrays that NumPy can cast safely to float64 (float32, integers, Fortran order, strided views) are converted to a C-contiguous float64 copy first; others (complex, strings) raise `TypeError`. Source indices are int64, converted the same way (float indices raise `TypeError`).
- Results are new float64 arrays; the C++ result is copied once into them.
- The GIL is released while a kernel runs, so Python threads can run kernels at the same time. Do not modify the inputs from another thread during a call.
- Invalid input raises `ValueError` with a message.

## Validation
| Check | Tolerance | Measured |
|---|---|---|
| FMM homogeneous, orders 1/2 (max relative error, round-off) | < 1e-11 | 2.0e-15 / 2.0e-15 (2-D and 3-D, 5 grid/source cases) |
| FMM vertical gradient, max-error ratio as h halves, order 1 | 1.9–2.1 | 2.001, 2.000 (2-D, h = 20, 10, 5 m); 2.019 (3-D, h = 25, 12.5 m) |
| FMM vertical gradient, order 2: max-error / RMS-error ratio | > 2.5 / > 3.2 | 2-D: 3.00, 2.81 / 3.60, 3.52; 3-D: 2.98 / 3.34 (max norm dominated by first-order fallback nodes; see docs/FMM.md, "Accuracy") |
| FMM vs Treister & Haber (2016) Table 2, orders 1/2, h = 1/40, 1/80 (max and RMS error) | within 1 unit of the 3rd printed digit | 7 of 8 values round to the published ones; order-1 max at h = 1/40: 2.651e-2 vs 2.66e-2 |
| FMM vs eikonalfm 0.9.8, orders 1/2: homogeneous, gradient 2-D and 3-D (max relative difference in T) | < 1e-11 | 6e-14 – 7e-13 (Linux), 6e-14 – 9e-13 (Windows) |
| FMM vs eikonalfm, smoothed Marmousi, surface and deep source, order 1 | < 1e-10 | 9.5e-12, 1.0e-11 (Linux); 7.9e-12, 9.9e-12 (Windows) |
| FMM vs eikonalfm, smoothed Marmousi, order 2 (max / median) | < 1e-5 / < 1e-11 | max 3.7e-7 and 3.9e-9 (Linux), 1.9e-6 and 5.2e-9 (Windows); median ≤ 1.2e-12. Round-off flips one comparison of nearly equal traveltimes; the nodes downstream differ (docs/FMM.md, "Comparison with eikonalfm") |
| NES `VerticalGradient.time()`, max-error ratio as h halves (20 → 10 m), orders 1/2 | 1.9–2.1 / > 2.5 | 2.001 / 3.000 |
| `fmm_multi` vs serial `fmm`, 1 and 4 threads (C++ and Python) | bitwise | equal |
| NES T vs Keras 3 float64 (max relative difference): 4 legacy models, fresh NES-OP and NES-TP, 9 activations, options off | < 1e-13 | 2.3e-16 – 8.0e-16 (Linux). Options off: relative to max \|T\|, because the T of that network crosses 0 |
| NES ∇T (reverse mode) vs Keras 3 float64 autodiff (max \|G − G_ref\| / max \|G_ref\|), same models | < 1e-13 | 2.6e-16 – 1.3e-15 (Linux) |
| NES ∇T vs the original TF2 NES in float32, 4 legacy models | < 1e-6 | 1.9e-7 – 3.2e-7 |
| NES ∇T vs central differences of T (C++, step 1e-6) | < 1e-8 | pass |
| NES eikonal residual \|∇T\| v − 1 on a 41 × 41 grid, C++ vs Keras (fresh NES-OP and NES-TP) | < 1e-12 | max difference 4.4e-16; mean \|r\| 0.0777 (OP), 0.0692 (TP) in both |
| NES T vs the original TF2 NES in float32 (`tests/data/legacy_refs.npz`), 4 legacy models | < 1e-6 | 1.8e-7 – 3.9e-7 (float32 rounding) |
| NES export refuses `improved_mlp` fixtures | ValueError | yes |
| NES training loss and its gradient vs JAX autodiff of the Keras 3 float64 model (relative; gradient scaled by its largest entry): 5 networks, NES-OP and NES-TP, adaptive and fixed activations, options off | < 1e-13 | loss 1.1e-16 – 3.8e-16, gradient 5.4e-16 – 1.6e-15 (Linux) |
| NES training loss gradient vs central differences of the loss (C++, 12 networks) | < 1e-8 | pass |
| Two Adam steps vs the formula in `docs/NES_TRAINING.md` (C++) | < 1e-12 | pass |
| 10 full-batch training steps (256 points) vs Keras 3 float64 from the same start, weights (max difference / max \|weight\|), NES-OP and NES-TP | < 1e-13 | 1.6e-16, 1.4e-16 (Linux). With 300 points Keras differs by up to 1e-11: it rounds 1/N to float32 (`docs/NES_TRAINING.md`, "Comparison with Keras") |
| Training with a fixed seed, 1 vs 4 threads (C++) | bitwise | equal |
| Weights trained and saved by nes_cpp, read back into NES (`tools/import_nes_weights.py`): weights, and T (max relative difference) | bitwise / < 1e-13 | pass |

## Benchmarks
`python NES/cpp/benchmarks/bench_fmm.py`: medians of 3–21 runs. Velocity: 2 km/s plus a vertical gradient (1/s) and a lateral ripple, h = 10 m, source at the top centre. eikonalfm is timed for its solver call only (NES then multiplies by `eikonalfm.distance`); `nes_cpp` returns T.

Hardware: laptop with Intel Core i7-11370H (4 cores, 8 threads) and 24 GB RAM. Linux: WSL2 Ubuntu 24.04, GCC 13.3, eikonalfm 0.9.8 manylinux wheel. Windows 11: MSVC 19.29, eikonalfm 0.9.8 win_amd64 wheel. Python 3.12.

**One source, one thread** (time in s):

| Grid | Order | Linux: eikonalfm | Linux: nes_cpp | Windows: eikonalfm | Windows: nes_cpp |
|---|---|---|---|---|---|
| 201 × 201 | 2 | 0.010 | 0.011 | 0.022 | 0.011 |
| 501 × 501 | 2 | 0.080 | 0.078 | 0.150 | 0.078 |
| 1001 × 1001 | 1 | 0.385 | 0.344 | 0.692 | 0.342 |
| 1001 × 1001 | 2 | 0.394 | 0.342 | 0.698 | 0.340 |
| 101 × 101 × 101 | 2 | 0.686 | 0.653 | 1.249 | 0.666 |

On Linux a single solve is as fast as eikonalfm, which is also C++ (0.9–1.15×; slower on small grids, where the input check and the result copy count). nes_cpp runs equally fast on both systems; eikonalfm's Windows wheel is about 2 times slower than its Linux wheel, so on Windows nes_cpp is 1.9–2.0 times faster.

**32 sources on 501 × 501, order 2** (time in s, speed-up over the eikonalfm loop):

| Method | Threads | Linux | Windows |
|---|---|---|---|
| eikonalfm, Python loop | 1 | 2.36 (1.0) | 4.66 (1.0) |
| `nes_cpp.fmm`, Python loop | 1 | 2.34 (1.0) | 2.53 (1.8) |
| `nes_cpp.fmm_multi` | 1 | 2.46 (0.96) | 2.45 (1.9) |
| `nes_cpp.fmm_multi` | 2 | 1.29 (1.8) | 1.27 (3.7) |
| `nes_cpp.fmm_multi` | 4 | 0.73 (3.2) | 0.70 (6.6) |
| `nes_cpp.fmm_multi` | 8 | 0.49 (4.8) | 0.52 (9.0) |

The gain comes from running sources in parallel: 3.2–3.4 times faster with 4 threads on 4 cores, and about 1.5 times more from hyper-threading.

**NES inference** (`python NES/cpp/benchmarks/bench_nes.py`, Linux only): NES-OP with the NES defaults (4 hidden layers × 50, untrained) and the local NES-TP model `tests/Models/Marmousi` (4 × 50, reciprocity `'output'`); points uniform in the model box. Keras runs `nes.Traveltime` and `nes.predict(x, ('T', 'G'))` (or `('T', 'Gr', 'Gs')`) with NES's default batch of 100000 points; nes_cpp calls `traveltime` and then `gradient` (which computes T again). nes_cpp is float64; *portable* is the default pip build (SSE2), *native* is built for this CPU with `-march=native` (AVX2, FMA, AVX-512; see "Build and test"). All threads unless noted. Keras versions: keras 3.15.1, jax 0.12.0.1, tensorflow-cpu 2.21.0, torch 2.15.0+cpu. All columns come from one run of the script.

Seconds per call (median of 5 runs for N = 10⁴, 1 run for 10⁷):

| Model, quantity | N | nes_cpp portable | nes_cpp native | JAX f32 | JAX f64 | TF f32 | TF f64 | PyTorch f32 | PyTorch f64 |
|---|---|---|---|---|---|---|---|---|---|
| NES-OP, T | 10⁴ | 0.0058 | 0.0013 | 0.011 | 0.023 | 0.065 | 0.051 | 0.0094 | 0.024 |
| NES-OP, T | 10⁷ | 4.86 | 1.48 | 2.51 | 4.52 | 7.04 | 16.7 | 8.99 | 20.9 |
| NES-OP, T + ∇T | 10⁴ | 0.013 | 0.0036 | 0.034 | 0.056 | 0.054 | 0.070 | 0.017 | 0.031 |
| NES-OP, T + ∇T | 10⁷ | 13.6 | 4.08 | 4.87 | 11.2 | 16.1 | 36.6 | 17.7 | 37.1 |
| NES-TP, T | 10⁴ | 0.0098 | 0.0027 | 0.021 | 0.041 | 0.047 | 0.067 | 0.016 | 0.034 |
| NES-TP, T | 10⁷ | 9.87 | 2.93 | 4.58 | 9.41 | 14.0 | 32.7 | 16.1 | 32.4 |
| NES-TP, T + ∇T | 10⁴ | 0.026 | 0.0079 | 0.068 | 0.117 | 0.067 | 0.106 | 0.030 | 0.068 |
| NES-TP, T + ∇T | 10⁷ | 26.3 | 8.45 | 11.5 | 22.5 | 33.9 | 66.8 | 34.3 | 74.5 |

What this shows:
- **Small batches (10³–10⁵ points), portable build:** nes_cpp is faster than every backend for T and for T + ∇T (1.1 to 77 times), since there is no framework dispatch per call (TensorFlow has a fixed cost of about 40 ms per `predict`).
- **Large batches (10⁶–10⁷ points), portable build:** as fast as JAX in float64 (0.8–1.2×) and 2.4–4.3 times faster than TensorFlow and PyTorch in float64, but 2 to 2.8 times slower than JAX in float32. XLA compiles JAX for this CPU's vector units, while the portable build uses only SSE2.
- **Native build, 10⁷ points:** T is 1.6–1.7 times faster than JAX float32 and 3.1–3.2 times faster than JAX float64; T + ∇T is 1.2–1.4 times faster than JAX float32 and 2.7 times faster than JAX float64.
- **Gradient cost:** the reverse-mode gradient call costs 1.7–1.9 traveltime calls (T + ∇T minus T), whatever the number of inputs. A forward-mode version, which costs 1 + n_inputs passes, was measured too: reverse mode makes NES-TP T + ∇T at 10⁷ points 1.8 times faster (2.3 times with the native build), and NES-OP 1.25 times (1.6).
- **Batch size:** 128 points per batch keep a batch's arrays in the L2 cache; with 512 points the NES-TP gradient (then in forward mode) ran 3 times slower, while T did not change.
- **Threads:** NES-OP T at 10⁷ takes 19.5 s on one thread and 4.86 s on all 8, a 4.0× speed-up on 4 cores.

The full tables for N = 10³ … 10⁷ and both thread counts are printed by the script.

**NES training** (`python NES/cpp/benchmarks/bench_train.py`, Linux only): seconds per epoch, median of 3 runs of 20 epochs, each after one warm-up epoch (which also compiles the Keras training step). Three cases, each starting from the same exported network in every column: the gallery network of the main README (NES-OP 5 × 32, 4000 points, batches of 500, lr 6e-3, decay 3e-4), and the NES defaults (4 × 50, 20000 points, batches of 2000, NES's default lr and decay) for NES-OP and NES-TP. Keras runs `nes.train` after NES's default `compile`; same versions and builds as above.

| Case | nes_cpp portable | nes_cpp portable, 1 thread | nes_cpp native | JAX f32 | JAX f64 | TF f32 | TF f64 | PyTorch f32 | PyTorch f64 |
|---|---|---|---|---|---|---|---|---|---|
| gallery NES-OP 5 × 32 | 0.0129 | 0.0346 | 0.0063 | 0.0069 | 0.0154 | 0.046 | 0.063 | 0.046 | 0.064 |
| NES-OP 4 × 50 | 0.099 | 0.256 | 0.034 | 0.044 | 0.089 | 0.124 | 0.194 | 0.105 | 0.174 |
| NES-TP 4 × 50 | 0.162 | 0.468 | 0.088 | 0.087 | 0.218 | 0.204 | 0.339 | 0.174 | 0.336 |

What this shows:
- **Portable build:** trains about as fast as JAX in float64 (0.9–1.35×) and 1.8–5.0 times faster than TensorFlow and PyTorch in float64; JAX in float32 is about 2 times faster.
- **Native build:** as fast as JAX in float32 or faster (1.0–1.3×) while computing in float64, and 2.5 times faster than JAX in float64.
- **Threads:** 2.6–2.9 times faster on all 8 threads than on one (4 cores). A gallery batch of 500 points is 4 chunks of 128, so at most 4 threads work on it.
- Every run trains: the final losses are 1e-2 to 2e-2 in every column. They are not equal, because each framework shuffles the points differently.

## Scale-out
The demos in `examples/` need `pip install "dask[array]" ray`; measured on the same laptop (WSL2, 8 logical CPUs), with Dask 2026.8.0 and Ray 2.58.1.

- **Threads** (`threads_demo.py`): 32 FMM solves on a 501 × 501 grid from a `ThreadPoolExecutor`. nes_cpp releases the GIL during the solve, so the threads run in parallel: 1.73 s with 1 thread, 0.34 s with 8 (5.1×), and the results equal `fmm_multi` bit for bit. eikonalfm keeps the GIL, so its threads take turns: 1.52 s with 1 thread, 1.61 s with 8.
- **Dask** (`dask_demo.py`): 2·10⁷ receivers as a lazy `dask.array` in 20 chunks of 10⁶ rows; `map_blocks` applies `NesModel.traveltime` to every chunk, and the mean and a histogram of T come out of one streaming pass, without keeping all traveltimes in memory. Times: 10.5 s with Dask's synchronous scheduler (OpenMP on all cores inside each chunk), 11.2 s with Dask threads (one chunk per core, OpenMP set to one thread to avoid oversubscription), 19.6 s with Dask processes (the chunks travel between processes). All three give the same numbers. The model is loaded once per process from its file path, because a `NesModel` cannot be pickled; the same code runs on a `dask.distributed` cluster.
- **Ray** (`ray_demo.py`): the smoothed Marmousi model is put once into Ray's object store (`ray.put`); 16 tasks of 16 sources each run `fmm_multi` on one core. 256 sources (783 MB of tables): 5.52 s with Ray, 5.25 s with one `fmm_multi` call (OpenMP); 64 sources: 2.12 s against 1.21 s. Ray has a fixed cost of about 1 s here (starting tasks, moving the tables through the object store), so each task needs enough work. The tables equal `fmm_multi` bit for bit.

**When Dask, when Ray.** Dask is for data parallelism over chunked collections: the same operation on every chunk of an array or a dataframe, built as a lazy graph, with streaming reductions and data larger than memory. NES inference over many points is that case. Ray is for task parallelism: independent, possibly different tasks (and stateful actors) that share large inputs through an object store. FMM tables, one task per batch of sources with the velocity shared once, are that case. On one machine both cost more than threads or OpenMP inside nes_cpp; they pay off when the work spans several machines, or does not fit in one process.

## Known limitations
- Point sources on grid nodes; regular grids; isotropic media.
- A single FMM solve is sequential; parallelism is across sources (FMM) or points (NES).
- NES inference and training support dense `TraveltimeNet` models with reciprocity `None` or `'output'` and the activations listed above. The NES parity tests need NES with Keras and a backend; they run in CI and in WSL, not in the Windows environment used here.
- Training is the plain NES recipe only: the NES loss (mean absolute eikonal residual), Adam with inverse-time decay, shuffled mini-batches. No callbacks (early stopping, RAR), sample weights or other optimizers; a different optimizer can use `loss_and_gradient` from Python.

## References
- Treister, E., & Haber, E. (2016). A fast marching algorithm for the factored eikonal equation. *J. Comput. Phys.*, 324. arXiv:1607.00973
- Grubas, S., Duchkov, A., & Loginov, G. (2023). Neural Eikonal Solver: Improving accuracy of physics-informed neural networks for solving eikonal equation in case of caustics. *J. Comput. Phys.*, 474, 111789.
