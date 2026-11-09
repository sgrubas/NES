"""nes_cpp.fmm and nes_cpp.fmm_multi from Python: results, array handling, errors, threads."""
import os
import subprocess
import sys

import numpy as np
import pytest

import nes_cpp


def smooth_velocity(shape, h):
    """2 km/s plus a vertical gradient and a lateral ripple, on a grid with node 0 at the origin."""
    axes = [np.arange(n) * dh for n, dh in zip(shape, h)]
    x = np.meshgrid(*axes, indexing="ij")
    return 2.0 + x[-1] + 0.3 * np.sin(10.0 * x[0])


@pytest.mark.parametrize("order", [1, 2])
def test_homogeneous_medium_is_exact(order):
    v = np.full((41, 31), 2.5)
    h = (0.01, 0.015)
    source = (17, 9)
    x, z = np.meshgrid(np.arange(41) * h[0], np.arange(31) * h[1], indexing="ij")
    exact = np.hypot(x - source[0] * h[0], z - source[1] * h[1]) / 2.5

    T = nes_cpp.fmm(v, source, h, order)
    assert T.shape == v.shape and T.dtype == np.float64
    # rtol only: T must be exactly 0 at the source
    np.testing.assert_allclose(T, exact, rtol=1e-11, atol=0)


def test_order_defaults_to_2():
    v = smooth_velocity((30, 20), (0.05, 0.05))
    np.testing.assert_array_equal(nes_cpp.fmm(v, (3, 4), (0.05, 0.05)),
                                  nes_cpp.fmm(v, (3, 4), (0.05, 0.05), 2))


def test_safely_castable_arrays_give_the_same_result():
    v = smooth_velocity((30, 20), (0.05, 0.05))
    args = ((3, 4), (0.05, 0.05))
    reference = nes_cpp.fmm(v, *args)

    # Converted to a C-contiguous float64 copy before the call: same values, same result.
    np.testing.assert_array_equal(nes_cpp.fmm(np.asfortranarray(v), *args), reference)
    padded = np.zeros((60, 20))
    padded[::2] = v
    np.testing.assert_array_equal(nes_cpp.fmm(padded[::2], *args), reference)
    v32 = v.astype(np.float32)
    np.testing.assert_array_equal(nes_cpp.fmm(v32, *args),
                                  nes_cpp.fmm(v32.astype(np.float64), *args))
    np.testing.assert_array_equal(nes_cpp.fmm(np.full((30, 20), 2), *args),
                                  nes_cpp.fmm(np.full((30, 20), 2.0), *args))


def test_invalid_input_raises():
    v = np.ones((5, 4))
    with pytest.raises(ValueError, match="2 or 3 axes"):
        nes_cpp.fmm(np.ones(5), (1,), (0.1,))
    with pytest.raises(ValueError, match="spacing"):
        nes_cpp.fmm(v, (1, 1), (0.1,))
    with pytest.raises(ValueError, match="outside the grid"):
        nes_cpp.fmm(v, (5, 0), (0.1, 0.1))
    with pytest.raises(ValueError, match="order"):
        nes_cpp.fmm(v, (1, 1), (0.1, 0.1), 3)
    with pytest.raises(ValueError, match="velocity"):
        nes_cpp.fmm(-v, (1, 1), (0.1, 0.1))
    with pytest.raises(TypeError):  # complex cannot be cast safely to float64
        nes_cpp.fmm(v + 1j, (1, 1), (0.1, 0.1))

    with pytest.raises(ValueError, match="shape"):
        nes_cpp.fmm_multi(v, np.array([1, 1]), (0.1, 0.1))
    with pytest.raises(TypeError):  # float indices cannot be cast safely to int64
        nes_cpp.fmm_multi(v, np.array([[1.0, 1.0]]), (0.1, 0.1))


def test_fmm_multi_shape():
    v = smooth_velocity((30, 20), (0.05, 0.05))
    table = nes_cpp.fmm_multi(v, [[0, 0], [29, 19], [10, 5]], (0.05, 0.05))
    assert table.shape == (3, 30, 20) and table.dtype == np.float64
    assert nes_cpp.fmm_multi(v, np.zeros((0, 2), dtype=np.int64), (0.05, 0.05)).shape == (0, 30, 20)


# Runs fmm_multi in a fresh process, so that OMP_NUM_THREADS is read when OpenMP starts.
RUN_FMM_MULTI = """
import sys
import numpy as np
import nes_cpp
v, sources = np.load(sys.argv[1]), np.load(sys.argv[2])
np.save(sys.argv[3], nes_cpp.fmm_multi(v, sources, (0.01, 0.01), int(sys.argv[4])))
"""


@pytest.mark.parametrize("order", [1, 2])
def test_fmm_multi_equals_serial_fmm_for_any_thread_count(tmp_path, order):
    v = smooth_velocity((121, 81), (0.01, 0.01))
    sources = np.array([[0, 0], [60, 40], [120, 10], [33, 80], [90, 25], [5, 70]])
    serial = np.stack([nes_cpp.fmm(v, s, (0.01, 0.01), order) for s in sources])

    np.save(tmp_path / "v.npy", v)
    np.save(tmp_path / "sources.npy", sources)
    for threads in ("1", "4"):
        out = tmp_path / f"table_{threads}.npy"
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        subprocess.run([sys.executable, "-c", RUN_FMM_MULTI, str(tmp_path / "v.npy"),
                        str(tmp_path / "sources.npy"), str(out), str(order)], env=env, check=True)
        assert np.array_equal(np.load(out), serial), f"{threads} threads"
