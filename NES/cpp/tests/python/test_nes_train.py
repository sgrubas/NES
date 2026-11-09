"""nes_cpp training against NES (docs/NES_TRAINING.md): the loss and its gradient against JAX autodiff,
trained weights read back by NES, and training against Keras training from the same start."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import nes_cpp

os.environ.setdefault("KERAS_BACKEND", "jax")
NES = pytest.importorskip("NES")  # needs Keras and a backend
import keras  # noqa: E402
from NES.utils import Uniform_PDF  # noqa: E402
from NES.velocity import VerticalGradient  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
sys.path.insert(0, str(HERE))
import nes_reference  # noqa: E402
from export_nes_weights import export  # noqa: E402
from import_nes_weights import load_into  # noqa: E402

RNG = np.random.default_rng(0)
VEL = VerticalGradient(v0=2.0, a=1.0, xmin=[0.0, 0.0], xmax=[1.0, 1.0])

# name: (solver, build_model arguments)
CASES = {
    "NES-OP, adaptive gauss": ("OP", dict(nl=3, nu=16)),
    "NES-OP, fixed tanh, unequal widths": ("OP", dict(nl=2, nu=[12, 7], act="tanh", out_act="sigmoid")),
    "NES-TP, reciprocity output": ("TP", dict(nl=3, nu=16, reciprocity="output")),
    "NES-TP, no reciprocity, swish": ("TP", dict(nl=2, nu=10, act="ad-swish-2", reciprocity=None)),
    "NES-TP, options off": ("TP", dict(nl=2, nu=10, act="ad-sin-1", out_act="linear", factored=False,
                                       out_vscale=False, input_scale=False, reciprocity=None)),
}


def build(solver, seed=1, **arguments):
    keras.utils.set_random_seed(seed)
    nes = NES.NES_OP(xs=[0.3, 0.2], velocity=VEL) if solver == "OP" else NES.NES_TP(VEL)
    nes.build_model(**arguments)
    return nes


def cpp_model(nes, path):
    export(nes, path)
    return nes_cpp.NesModel(path)


@pytest.mark.parametrize("name", list(CASES))
def test_loss_and_gradient_match_jax(name, tmp_path, float64):
    solver, arguments = CASES[name]
    nes = build(solver, **arguments)
    for layer in nes.net.acts + [nes.net.out_act]:
        if layer.adapt:
            layer.a.assign([0.9])  # away from 1, so that the a-derivatives are tested
    model = cpp_model(nes, tmp_path / "model.nesw")
    x = RNG.uniform(0, 1, (300, model.n_inputs))
    v = nes_reference.velocities_at_receivers(nes, x).reshape(-1)

    assert np.array_equal(model.parameters(), nes_reference.parameters(nes))
    loss_ref, gradient_ref = nes_reference.loss_and_gradient(nes, x, v)
    loss, gradient = model.loss_and_gradient(x, v)
    difference = np.abs(gradient - gradient_ref).max() / np.abs(gradient_ref).max()
    print(f"{name}: loss {abs(loss - loss_ref) / loss_ref:.1e}, gradient {difference:.1e}")
    assert abs(loss - loss_ref) / loss_ref < 1e-13
    assert difference < 1e-13


@pytest.mark.parametrize("solver", ["OP", "TP"])
def test_trained_weights_read_back_by_nes(solver, tmp_path, float64):
    nes = build(solver, nl=2, nu=12)
    model = cpp_model(nes, tmp_path / "start.nesw")
    x = RNG.uniform(0, 1, (500, model.n_inputs))
    model.train(x, nes_reference.velocities_at_receivers(nes, x).reshape(-1), epochs=3, batch_size=100)
    model.save(tmp_path / "trained.nesw")
    load_into(nes, tmp_path / "trained.nesw")

    assert np.array_equal(nes_reference.parameters(nes), model.parameters())
    points = RNG.uniform(0, 1, (200, model.n_inputs))
    T, T_nes = model.traveltime(points), nes.Traveltime(points)
    assert np.max(np.abs(T - T_nes) / np.abs(T_nes)) < 1e-13


@pytest.mark.parametrize("solver", ["OP", "TP"])
def test_training_steps_match_keras(solver, tmp_path, float64):
    """Ten Adam steps from the same start give the weights Keras gives. One batch with all points,
    so that their order does not matter; 256 points, because Keras scales the batch gradient by 1/N
    rounded to float32, which is exact for N = 256; a short run, because training amplifies rounding
    differences (docs/NES_TRAINING.md, "Comparison with Keras")."""
    nes = build(solver, nl=3, nu=16)
    nes.compile(lr=6e-3, decay=3e-4)
    model = cpp_model(nes, tmp_path / "start.nesw")
    x = Uniform_PDF(VEL)(256, rep=model.n_inputs // 2)
    v = nes_reference.velocities_at_receivers(nes, x).reshape(-1)

    nes.train(x, batch_size=256, epochs=10, shuffle=False, verbose=0)
    history = model.train(x, v, batch_size=256, epochs=10, learning_rate=6e-3, decay=3e-4)
    p, p_keras = model.parameters(), nes_reference.parameters(nes)
    difference = np.abs(p - p_keras).max() / np.abs(p_keras).max()
    print(f"NES-{solver}: loss {history[0]:.3e} -> {history[-1]:.3e}; weights after 10 steps, "
          f"C++ vs Keras: {difference:.1e}")
    assert history[-1] < 0.5 * history[0]
    assert difference < 1e-13


def test_train_options(tmp_path):
    nes = build("TP", nl=2, nu=8)
    x = RNG.uniform(0, 1, (300, 4))
    v = np.full(300, 2.0)
    first = cpp_model(nes, tmp_path / "a.nesw")
    second = cpp_model(nes, tmp_path / "b.nesw")
    history = first.train(x, v, epochs=2, batch_size=64)  # NES's default rate for NES_TP: 5e-3
    assert history.shape == (2,)
    np.testing.assert_array_equal(history, second.train(x, v, epochs=2, batch_size=64, learning_rate=5e-3))
    np.testing.assert_array_equal(first.parameters(), second.parameters())
    with pytest.raises(ValueError):
        first.train(x, v[:10])
    with pytest.raises(ValueError):
        first.set_parameters(np.zeros(3))
