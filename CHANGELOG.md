# Changelog

All notable changes to NES are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-09-30

NES now runs on [Keras 3](https://keras.io/keras_3/) with the **JAX**, **TensorFlow** or **PyTorch** backend.
The public API of 0.2 is unchanged, and models saved by 0.2 still load.

### Changed
- Ported from TensorFlow 2 / Keras 2 to Keras 3. `NES/backend.py` holds the per-backend input gradients
  (`jax.vjp`, `tf.GradientTape`, `torch.autograd`); everything else uses `keras.ops`. This replaces
  `tf.gradients` inside `Lambda` layers, which cannot work on JAX or PyTorch.
- `NES_OP` and `NES_TP` share one base class. Inputs are one `(N, dim)` tensor per point instead of one
  `Input` per coordinate. Outputs are built lazily, and `predict(x, ('T', 'Gs'))` returns several outputs
  from one pass.
- The traveltime network is a subclassed `keras.Model` (`NES.layers.TraveltimeNet`) with the improved
  factorization, input scaling and `improved_mlp`.
- Custom eikonal layers receive the gradient as one `(N, dim)` tensor instead of a list of `(N, 1)` tensors.
- Learning-rate decay is inverse-time decay again. On TensorFlow >= 2.11 the old code passed `decay` as
  decoupled weight decay.
- `setup.py` no longer imports NES at install time. Requirements are `numpy`, `scipy`, `h5py` and
  `keras>=3.8`, plus one backend through the extras `jax`, `tensorflow` or `torch` (and `hpo`, `test`).
- The tutorials use `keras.backend.clear_session()` instead of `tf.keras.backend.clear_session()`.

### Added
- `reciprocity` option of `NES_TP.build_model`: `'output'` (default, eq. 10 of the paper), `'first_layer'`
  and `'invariant'`. The last two train 1.7-2.4x faster per epoch; `'output'` was the most accurate for the
  same number of epochs on the Luneburg lens. `benchmarks/reciprocity.py` compares them.
- `NES.legacy`: models saved by NES <= 0.2 load and reproduce the original outputs to float32 precision,
  including the optimizer state.
- Analytic test models with closed-form two-point traveltimes: `NES.velocity.LuneburgLens` (low or high
  velocity) and `NES.velocity.MaxwellFishEye` (with its high-velocity hyperbolic twin).
- `NES.hpo`: hyperparameter search with Optuna >= 5 (two objectives, validation loss and training FLOPs;
  median stopping rule; PED-ANOVA importance). Tutorial: `notebooks/NES_HPO_Optuna.ipynb`.
- Test suite (`tests/`) and GitHub Actions CI on all three backends.
- `benchmarks/analytic_models.py` (the README gallery) and `benchmarks/reciprocity.py`.

### Fixed
- float64 mode: traveltimes were limited to about 5e-8 relative error on JAX and PyTorch. Keras 3.15
  computes `ops.norm`, trigonometric and hyperbolic functions in float32 for float64 input on these backends,
  and NES itself cast the inputs, the velocities and the source position to float32. NES now keeps float64
  throughout when `floatx` is float64 (see the README for how to enable it).
- At the source, the traveltime and its gradient are 0 on every backend; the gradient was NaN on JAX and
  TensorFlow.
- `VerticalGradient.dtime` used `a*2` instead of `a**2`.
- `LocAnomaly.gradient` divided by `sigma` instead of `sigma**2`.
- `Generator` shuffled the points but not their weights, and built `add_data` weights of the wrong length.
- `NES_TP.load` ignored the saved losses; `BestWeights` printed an undefined attribute.
- `Interpolator` pickles survive SciPy upgrades.
- `np.trapz` (removed in NumPy 2) in ray tracing; `sinc` had a NaN gradient at 0.

### Removed
- The pin to TensorFlow <= 2.15 and the dependency on Keras 2.

## [0.2.2] and earlier

TensorFlow 2 / Keras 2 implementation accompanying the paper
([J. Comput. Phys. 474, 111789, 2023](https://doi.org/10.1016/j.jcp.2022.111789)). See the git history.

[0.3.0]: https://github.com/sgrubas/NES/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/sgrubas/NES/releases/tag/v0.2.2
