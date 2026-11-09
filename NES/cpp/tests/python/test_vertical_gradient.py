"""nes_cpp.fmm on NES.velocity.VerticalGradient against its analytic time().

Checks that the grid conventions match NES (coordinates (x, z) with z last, meshgrid indexing 'ij',
node 0 at the origin) and that the errors converge: when h halves, the max error halves for order 1
and drops by about 3 for order 2 (docs/FMM.md, "Accuracy").
"""
import os

import numpy as np
import pytest

import nes_cpp

os.environ.setdefault("KERAS_BACKEND", "jax")
velocity = pytest.importorskip("NES.velocity")  # importing NES needs Keras and a backend


def max_error(model, source, h, order):
    """Largest |T - T_exact| on [0, 2] x [0, 1] km with spacing h."""
    axes = [np.arange(round(2.0 / h) + 1) * h, np.arange(round(1.0 / h) + 1) * h]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)  # (n_x, n_z, 2), z last
    index = tuple(round(c / h) for c in source)
    T = nes_cpp.fmm(model(X), index, (h, h), order)
    T_exact = model.time(X.reshape(-1, 2), np.asarray(source)).reshape(T.shape)
    return np.abs(T - T_exact).max()


@pytest.mark.parametrize("order, low, high", [(1, 1.9, 2.1), (2, 2.5, np.inf)])
def test_error_ratio_when_h_halves(order, low, high):
    model = velocity.VerticalGradient(v0=2.0, a=1.0)  # v = 2 km/s + z * 1/s
    source = (0.5, 0.2)  # km, a node on both grids
    ratio = max_error(model, source, 0.02, order) / max_error(model, source, 0.01, order)
    print(f"order {order}: max-error ratio {ratio:.3f}")
    assert low < ratio < high
