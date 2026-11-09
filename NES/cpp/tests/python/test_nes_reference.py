"""Python side of the C++ training port: weight export/import round trip and the JAX reference of the
training loss and its gradient (nes_reference.py)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
NES = pytest.importorskip("NES")  # needs Keras and a backend
import keras  # noqa: E402
from NES.velocity import VerticalGradient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_nes_weights import export  # noqa: E402
from import_nes_weights import load_into, read_weights  # noqa: E402
import nes_reference  # noqa: E402

RNG = np.random.default_rng(0)


@pytest.fixture
def vel():
    return VerticalGradient(v0=2.0, a=1.0, xmin=[0.0, 0.0], xmax=[1.0, 1.0])


def make(kind, vel, seed, reciprocity="output", **build):
    """A NES_OP / NES_TP with random weights; `seed` changes the weights."""
    keras.utils.set_random_seed(seed)
    nes = NES.NES_OP(xs=[0.3, 0.2], velocity=vel) if kind == "OP" else NES.NES_TP(vel)
    options = dict(nl=3, nu=[7, 9, 5]) | build
    if kind == "TP":
        options["reciprocity"] = reciprocity
    nes.build_model(**options)
    # make every adaptive `a` differ from its initial value 1
    for layer in nes.net.acts + [nes.net.out_act]:
        if layer.adapt:
            layer.a.assign(RNG.uniform(0.6, 1.4, (1,)))
    return nes


def points(nes, n=200):
    return RNG.uniform(0.05, 1.0, (n, nes._n_in))


CASES = [("OP", None, "ad-gauss-1", "ad-sigmoid-1"),
         ("OP", None, "tanh", "sigmoid"),
         ("TP", "output", "ad-tanh-2", "ad-sigmoid-1"),
         ("TP", None, "ad-sin-1", "sigmoid"),
         ("TP", "output", "gauss", "ad-sigmoid-1")]


@pytest.mark.parametrize("kind, reciprocity, act, out_act", CASES)
def test_export_import_round_trip(kind, reciprocity, act, out_act, tmp_path, float64, vel):
    a = make(kind, vel, 1, reciprocity, act=act, out_act=out_act)
    b = make(kind, vel, 2, reciprocity, act=act, out_act=out_act)
    x = points(a)
    assert not np.allclose(a.Traveltime(x), b.Traveltime(x))

    export(a, tmp_path / "a.nesw")
    load_into(b, tmp_path / "a.nesw")
    np.testing.assert_array_equal(nes_reference.parameters(b), nes_reference.parameters(a))
    np.testing.assert_array_equal(b.Traveltime(x), a.Traveltime(x))

    header, arrays = read_weights(tmp_path / "a.nesw")
    assert header["widths"] == [a.net.n_features, 7, 9, 5]
    assert [w.shape for w in arrays[:2]] == [(a.net.n_features, 7), (7,)]


def test_import_refuses_other_architectures(tmp_path, float64, vel):
    export(make("TP", vel, 1, "output"), tmp_path / "tp.nesw")
    export(make("OP", vel, 1), tmp_path / "op.nesw")
    cases = [
        (make("TP", vel, 2, "output", nl=2, nu=[7, 9]), "tp.nesw", "widths"),
        (make("TP", vel, 2, "output", nu=[7, 9, 6]), "tp.nesw", "widths"),
        (make("TP", vel, 2, None), "tp.nesw", "reciprocity"),
        (make("TP", vel, 2, "output", act="tanh"), "tp.nesw", "hidden activation"),
        (make("TP", vel, 2, "output", act="ad-gauss-2"), "tp.nesw", "hidden activation"),
        (make("TP", vel, 2, "output", out_act="sigmoid"), "tp.nesw", "output activation"),
        (make("TP", vel, 2, "output", factored=False), "tp.nesw", "factored"),
        (make("TP", vel, 2, "output", out_vscale=False), "tp.nesw", "out_vscale"),
        (make("OP", vel, 2), "tp.nesw", "solver"),
        (make("TP", vel, 2, "output"), "op.nesw", "solver"),
    ]
    for nes, name, what in cases:
        with pytest.raises(ValueError, match=what):
            load_into(nes, tmp_path / name)
    other_source = NES.NES_OP(xs=[0.4, 0.2], velocity=vel)
    other_source.build_model(nl=3, nu=[7, 9, 5])
    with pytest.raises(ValueError, match="xs"):
        load_into(other_source, tmp_path / "op.nesw")


def test_import_refuses_truncated_file(tmp_path, float64, vel):
    nes = make("OP", vel, 1)
    export(nes, tmp_path / "m.nesw")
    raw = (tmp_path / "m.nesw").read_bytes()
    (tmp_path / "short.nesw").write_bytes(raw[:-8])
    with pytest.raises(ValueError, match="weights"):
        load_into(nes, tmp_path / "short.nesw")


@pytest.mark.parametrize("kind, reciprocity, act, out_act", CASES)
def test_reference_loss_equals_nes_loss(kind, reciprocity, act, out_act, float64, vel):
    nes = make(kind, vel, 3, reciprocity, act=act, out_act=out_act)
    x = points(nes)
    v = nes_reference.velocities_at_receivers(nes, x)
    loss, gradient = nes_reference.loss_and_gradient(nes, x, v)

    nes.train_inputs(x)
    nes.compile(jit_compile=False)
    nes_loss = nes.model.evaluate(nes.x_train, batch_size=len(x), verbose=0)
    # Keras's MAE loss and its reduction round through float32 (relative error ~1e-8) even in float64 mode;
    # the exact check is the hand computation below
    np.testing.assert_allclose(loss, nes_loss, rtol=1e-6)

    # by hand, from the gradient NES reports
    G = nes.Gradient(x) if kind == "OP" else nes.GradientR(x)
    hand = np.mean(np.abs((np.sum(G * G, axis=-1) * v[:, 0] ** 2 - 1) / 2))
    np.testing.assert_allclose(loss, hand, rtol=1e-12)
    assert gradient.shape == nes_reference.parameters(nes).shape


def set_parameter(nes, index, value):
    """Sets entry `index` of the flat parameter vector."""
    theta = nes_reference.parameters(nes)
    theta[index] = value
    start = 0
    for variable in nes_reference._variables_in_order(nes.net):
        size = int(np.prod(variable.shape))
        variable.assign(theta[start:start + size].reshape(variable.shape))
        start += size


def parameter_groups(nes):
    """Index of the first entry of each kind of parameter in the flat vector: name -> index."""
    net = nes.net
    groups, start = {}, 0
    for k, layer in enumerate(net.hidden + [net.head]):
        name = f"hidden{k}" if layer is not net.head else "head"
        groups[f"{name} kernel"] = start
        start += int(np.prod(layer.kernel.shape))
        groups[f"{name} bias"] = start
        start += int(np.prod(layer.bias.shape))
    if net.acts[0].adapt:
        groups["hidden a"] = start
        start += len(net.acts)
    if net.out_act.adapt:
        groups["output a"] = start
    return groups


@pytest.mark.parametrize("kind, reciprocity, act, out_act", CASES)
def test_reference_gradient_matches_central_differences(kind, reciprocity, act, out_act, float64, vel):
    nes = make(kind, vel, 4, reciprocity, act=act, out_act=out_act)
    x = points(nes)
    v = nes_reference.velocities_at_receivers(nes, x)
    loss, gradient = nes_reference.loss_and_gradient(nes, x, v)
    theta = nes_reference.parameters(nes)

    worst = 0.0
    for name, start in parameter_groups(nes).items():
        for index in (start, start + 3) if "kernel" in name else (start,):  # first and a later kernel entry
            h = 1e-6
            values = []
            for sign in (+1, -1):
                set_parameter(nes, index, theta[index] + sign * h)
                values.append(nes_reference.loss_and_gradient(nes, x, v)[0])
            set_parameter(nes, index, theta[index])
            numeric = (values[0] - values[1]) / (2 * h)
            difference = abs(numeric - gradient[index]) / max(abs(gradient[index]), 1e-3 * np.abs(gradient).max())
            print(f"{kind} {reciprocity} {name}[{index - start}]: AD {gradient[index]:.6e}, "
                  f"central {numeric:.6e}, relative {difference:.1e}")
            worst = max(worst, difference)
    print(f"largest relative difference {worst:.2e}")
    assert worst < 1e-6
    assert np.abs(gradient).max() > 0
