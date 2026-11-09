"""nes_cpp.NesModel against NES: exported legacy models (original TensorFlow/Keras-2 NES) and models
built with Keras 3 in float64, for T and its gradient (docs/NES_INFERENCE.md, docs/WEIGHTS_FORMAT.md)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import nes_cpp

os.environ.setdefault("KERAS_BACKEND", "jax")
NES = pytest.importorskip("NES")  # needs Keras and a backend
import keras  # noqa: E402
from NES.velocity import VerticalGradient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from export_nes_weights import SUPPORTED_ACTIVATIONS, export, load  # noqa: E402

DATA = Path(__file__).resolve().parents[4] / "tests" / "data"
REFS = dict(np.load(DATA / "legacy_refs.npz"))
SUPPORTED_LEGACY = ["OP_ad-gauss-1_imlpFalse", "OP_tanh_imlpFalse",
                    "TP_recipFalse_imlpFalse", "TP_recipTrue_imlpFalse"]  # reciprocity None / 'output'
RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def vel():
    return VerticalGradient(v0=2.0, a=1.0, xmin=[0.0, 0.0], xmax=[1.0, 1.0])


def cpp_model(nes, tmp_path):
    export(nes, tmp_path / "model.nesw")
    return nes_cpp.NesModel(tmp_path / "model.nesw")


def max_relative_difference(T, T_ref):
    """Largest |T - T_ref| / |T_ref|; where T_ref is 0 (the source of NES-OP), T must be 0 too."""
    T, T_ref = np.asarray(T).reshape(-1), np.asarray(T_ref).reshape(-1)
    zero = T_ref == 0
    assert np.all(T[zero] == 0)
    return np.max(np.abs(T[~zero] - T_ref[~zero]) / np.abs(T_ref[~zero]))


def max_scaled_difference(G, G_ref):
    """Largest |G - G_ref| / max |G_ref|: gradient components can be near 0, so scale by the largest."""
    return np.abs(np.asarray(G) - np.asarray(G_ref)).max() / np.abs(G_ref).max()


def compare_with_keras(nes, model, x, label, relative_T=True):
    """Max relative difference of T, and max scaled difference of the gradient(s), C++ vs Keras.
    relative_T=False: T as the gradients, scaled by max |T|, for networks whose T crosses 0."""
    T, T_ref = model.traveltime(x), nes.Traveltime(x)
    differences = {"T": max_relative_difference(T, T_ref) if relative_T else max_scaled_difference(T, T_ref)}
    if model.two_point:
        differences["Gr"] = max_scaled_difference(model.gradient_r(x), nes.GradientR(x))
        differences["Gs"] = max_scaled_difference(model.gradient_s(x), nes.GradientS(x))
    else:
        differences["G"] = max_scaled_difference(model.gradient(x), nes.Gradient(x))
    print(f"{label}: " + ", ".join(f"{k} {v:.2e}" for k, v in differences.items()))
    return max(differences.values())


def load_legacy(tag):
    cls = NES.NES_OP if tag.startswith("OP") else NES.NES_TP
    return cls.load(DATA / "legacy_models" / tag)


@pytest.mark.parametrize("tag", SUPPORTED_LEGACY)
def test_legacy_model_matches_original_float32_outputs(tag, tmp_path):
    """The references come from the original TF2 NES in float32: the difference is float32 rounding."""
    model = cpp_model(load_legacy(tag), tmp_path)
    x = REFS[f"{tag}/x"]
    difference = max_relative_difference(model.traveltime(x), REFS[f"{tag}/T"])
    gradient_keys = ["Gr", "Gs"] if model.two_point else ["G"]
    G = {"G": model.gradient(x), "Gr": model.gradient_r(x) if model.two_point else None,
         "Gs": model.gradient_s(x) if model.two_point else None}
    gradient_difference = max(max_scaled_difference(G[k], REFS[f"{tag}/{k}"]) for k in gradient_keys)
    print(f"{tag}: vs TF2 float32, T {difference:.2e}, gradient {gradient_difference:.2e}")
    assert difference < 1e-6
    assert gradient_difference < 1e-6


@pytest.mark.parametrize("tag", SUPPORTED_LEGACY)
def test_legacy_model_matches_keras_float64(tag, tmp_path, float64):
    nes = load_legacy(tag)  # float32 weights, evaluated in float64
    assert compare_with_keras(nes, cpp_model(nes, tmp_path), REFS[f"{tag}/x"], tag) < 1e-13


@pytest.mark.parametrize("tag", ["OP_ad-gauss-1_imlpTrue", "TP_recipTrue_imlpTrue"])
def test_improved_mlp_is_refused(tag, tmp_path):
    with pytest.raises(ValueError, match="improved_mlp not supported"):
        export(load_legacy(tag), tmp_path / "model.nesw")


@pytest.fixture
def trained(float64, vel):
    """A NES-OP and a NES-TP (reciprocity 'output'), trained for a few epochs in float64 so that the
    adaptive a move away from 1."""
    keras.utils.set_random_seed(5)
    op = NES.NES_OP(xs=[0.3, 0.2], velocity=vel)
    op.build_model(nl=3, nu=16)
    op.train(2000, epochs=3, batch_size=500, verbose=0)
    tp = NES.NES_TP(vel)
    tp.build_model(nl=3, nu=16)
    tp.train(2000, epochs=3, batch_size=500, verbose=0)
    return op, tp


def test_freshly_trained_float64_models(tmp_path, trained):
    op, tp = trained
    xr = np.vstack([RNG.uniform(0, 1, (1000, 2)), op.xs])  # with the source, where T = 0
    pairs = RNG.uniform(0, 1, (1000, 4))
    for name, nes, x in [("NES-OP", op, xr), ("NES-TP", tp, pairs)]:
        assert compare_with_keras(nes, cpp_model(nes, tmp_path), x, name) < 1e-13
    G_source = cpp_model(op, tmp_path).gradient(op.xs)
    assert np.array_equal(G_source, [0.0, 0.0])


def test_eikonal_residual_statistics(tmp_path, trained, vel):
    """|grad T| v - 1 on a grid: the same values, hence the same statistics, from C++ and Keras."""
    op, tp = trained
    axis = np.linspace(0.0, 1.0, 41)
    X = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    X = X[np.linalg.norm(X - op.xs, axis=-1) > 0]  # the gradient is set to 0 at the source
    v = vel(X)
    pairs = np.concatenate([np.full_like(X, 0.5), X], axis=-1)  # source at the centre
    pairs = pairs[np.linalg.norm(X - 0.5, axis=-1) > 0]
    v_pairs = vel(pairs[:, 2:])
    for name, residual, residual_ref in [
            ("NES-OP", np.linalg.norm(cpp_model(op, tmp_path).gradient(X), axis=-1) * v - 1,
             np.linalg.norm(op.Gradient(X), axis=-1) * v - 1),
            ("NES-TP", np.linalg.norm(cpp_model(tp, tmp_path).gradient_r(pairs), axis=-1) * v_pairs - 1,
             np.linalg.norm(tp.GradientR(pairs), axis=-1) * v_pairs - 1)]:
        print(f"{name}: mean |residual| C++ {np.abs(residual).mean():.6e}, Keras {np.abs(residual_ref).mean():.6e}; "
              f"max difference {np.abs(residual - residual_ref).max():.1e}")
        assert np.abs(residual - residual_ref).max() < 1e-12


@pytest.mark.parametrize("act", SUPPORTED_ACTIVATIONS)
def test_every_activation_matches_keras_float64(act, tmp_path, float64, vel):
    keras.utils.set_random_seed(6)
    nes = NES.NES_OP(xs=[0.3, 0.2], velocity=vel)
    nes.build_model(nl=2, nu=[12, 7], act=f"ad-{act}-2")  # unequal widths, n = 2
    for layer in nes.net.acts + [nes.net.out_act]:
        layer.a.assign([0.8])
    assert compare_with_keras(nes, cpp_model(nes, tmp_path), RNG.uniform(0, 1, (600, 2)), act) < 1e-13


def test_options_without_factorization_and_bounding(tmp_path, float64, vel):
    keras.utils.set_random_seed(7)
    nes = NES.NES_TP(vel)
    nes.build_model(nl=2, nu=10, act="tanh", out_act="linear", factored=False, out_vscale=False,
                    input_scale=False, reciprocity=None)
    x = RNG.uniform(0, 1, (300, 4))
    # Without the factor and with a linear output, T of the untrained network crosses 0. Near 0 a
    # relative difference is rounding divided by |T|, which grows without bound (1e-11 at
    # |T| = 2e-5 for this network), so T is compared relative to max |T| instead.
    assert compare_with_keras(nes, cpp_model(nes, tmp_path), x, "options off", relative_T=False) < 1e-13


def test_export_from_saved_folders(tmp_path, vel):
    """The command-line path: load a saved folder (Keras 3 or legacy format), then export."""
    keras.utils.set_random_seed(9)
    op = NES.NES_OP(xs=[0.3, 0.2], velocity=vel)
    op.build_model(nl=2, nu=8)
    op.save(str(tmp_path / "op"))
    tp = NES.NES_TP(vel)
    tp.build_model(nl=2, nu=8)
    tp.save(str(tmp_path / "tp"))
    for folder, nes, cls, n_in in [("op", op, NES.NES_OP, 2), ("tp", tp, NES.NES_TP, 4),
                                   (DATA / "legacy_models" / "TP_recipTrue_imlpFalse", None, NES.NES_TP, 4)]:
        loaded = load(tmp_path / folder)
        assert isinstance(loaded, cls)
        x = RNG.uniform(0, 1, (50, n_in))
        export(loaded, tmp_path / "loaded.nesw")
        T = nes_cpp.NesModel(tmp_path / "loaded.nesw").traveltime(x)
        if nes is not None:
            np.testing.assert_array_equal(T, cpp_model(nes, tmp_path).traveltime(x))


def test_points_shape(tmp_path, vel):
    keras.utils.set_random_seed(8)
    op = NES.NES_OP(xs=[0.3, 0.2], velocity=vel)
    op.build_model(nl=2, nu=8)
    model = cpp_model(op, tmp_path)
    assert (model.dim, model.two_point, model.n_inputs) == (2, False, 2)
    grid = RNG.uniform(0, 1, (5, 7, 2))
    assert model.traveltime(grid).shape == (5, 7)
    assert model.gradient(grid).shape == (5, 7, 2)
    np.testing.assert_array_equal(model.traveltime(grid).reshape(-1), model.traveltime(grid.reshape(-1, 2)))
    np.testing.assert_array_equal(model.gradient(grid).reshape(-1, 2), model.gradient(grid.reshape(-1, 2)))
    with pytest.raises(ValueError, match="shape"):
        model.traveltime(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="shape"):
        model.gradient(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="only for NES_TP"):
        model.gradient_r(grid)
    with pytest.raises(ValueError, match="cannot open"):
        nes_cpp.NesModel(tmp_path / "missing.nesw")

    tp = NES.NES_TP(vel)
    tp.build_model(nl=2, nu=8)
    model = cpp_model(tp, tmp_path)
    pairs = RNG.uniform(0, 1, (3, 4, 4))
    G = model.gradient(pairs)
    assert G.shape == (3, 4, 4)
    np.testing.assert_array_equal(model.gradient_s(pairs), G[..., :2])
    np.testing.assert_array_equal(model.gradient_r(pairs), G[..., 2:])
