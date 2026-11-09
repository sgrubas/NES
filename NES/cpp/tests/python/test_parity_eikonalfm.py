"""nes_cpp.fmm against eikonalfm.factored_fast_marching, the reference solver of NES (same scheme).

T is compared, never the factors: eikonalfm returns tau1 with T = tau1 * |x - x_s|. The source node
is left out of the relative difference: eikonalfm's distance there is about 1e-16 instead of 0.
"""
from pathlib import Path

import numpy as np
import pytest

import nes_cpp

eikonalfm = pytest.importorskip("eikonalfm")

MARMOUSI = Path(__file__).resolve().parents[3] / "data" / "Marmousi_Pwave_smooth_12_5m.npy"


def grid(shape, h):
    return np.meshgrid(*[np.arange(n) * dh for n, dh in zip(shape, h)], indexing="ij")


def marmousi():
    return np.load(MARMOUSI) / 1000.0  # m/s -> km/s, spacing 12.5 m


# name: (velocity, source index, spacing in km, tolerance on the max relative difference by order).
# The scheme is the same, so most nodes agree to round-off: the median relative difference must be
# below 1e-11 in every case. The max tolerances are about 10 times the largest value measured on
# Linux (GCC) and Windows (MSVC), README "Validation". In Marmousi at order 2, round-off flips a
# comparison of two nearly equal traveltimes at one node, and the nodes downstream of it differ by
# more (docs/FMM.md, "Comparison with eikonalfm"); which node flips depends on the compiler.
CASES = {
    "homogeneous 2-D": (lambda: np.full((201, 101), 2.0), (50, 20), (0.01, 0.01), {1: 1e-11, 2: 1e-11}),
    "gradient 2-D": (lambda: 2.0 + grid((201, 101), (0.01, 0.01))[1], (50, 20), (0.01, 0.01),
                     {1: 1e-11, 2: 1e-11}),
    "gradient 3-D": (lambda: 2.0 + grid((61, 41, 31), (0.02,) * 3)[2], (20, 20, 5), (0.02,) * 3,
                     {1: 1e-11, 2: 1e-11}),
    "Marmousi, surface source": (marmousi, (680, 0), (0.0125, 0.0125), {1: 1e-10, 2: 1e-5}),
    "Marmousi, deep source": (marmousi, (300, 200), (0.0125, 0.0125), {1: 1e-10, 2: 1e-5}),
}


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("name", list(CASES))
def test_matches_eikonalfm(name, order):
    make_v, source, h, tolerance = CASES[name]
    v = make_v()
    T = nes_cpp.fmm(v, source, h, order)
    T_ref = (eikonalfm.factored_fast_marching(v, source, h, order)
             * eikonalfm.distance(v.shape, h, source, indexing="ij"))

    assert T[source] == 0.0 and abs(T_ref[source]) < 1e-15
    others = T > 0
    relative_difference = np.abs(T[others] - T_ref[others]) / T_ref[others]
    print(f"{name}, order {order}: max relative difference {relative_difference.max():.2e}, "
          f"median {np.median(relative_difference):.1e}")
    assert np.median(relative_difference) < 1e-11
    assert relative_difference.max() < tolerance[order]
