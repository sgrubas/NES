"""Export a trained NES model to a weight file for nes_cpp.NesModel (docs/WEIGHTS_FORMAT.md).

    python NES/cpp/tools/export_nes_weights.py MODEL_DIR OUTPUT.nesw

MODEL_DIR is a folder written by NES_OP.save or NES_TP.save; models saved by the original
TensorFlow/Keras-2 NES load too (NES.legacy). From Python: `export(nes, path)` for a model in memory.
The weights are read from the network's variables, not from the files, so one exporter covers both
formats. Unsupported models (improved_mlp, reciprocity 'first_layer' or 'invariant', other
activations) raise ValueError.
"""
import os
import pathlib
import pickle
import sys

import numpy as np

FORMAT_VERSION = 1
SUPPORTED_ACTIVATIONS = ('gauss', 'tanh', 'sigmoid', 'sin', 'swish', 'linear', 'softplus', 'elu', 'relu')


def to_numpy(variable):
    """Values of a Keras variable as a NumPy array. Converts the variable's backend array: converting
    the variable itself triggers a NumPy 2 DeprecationWarning in Keras 3.15. Keras is imported here
    because load() chooses the backend first."""
    from keras import ops
    return np.asarray(ops.convert_to_numpy(variable.value))


def activation_spec(layer):
    """(name, n, a, adaptive) of an NES.layers.AdaptiveActivation, `act(x) = name(n * a * x)`."""
    spec = layer.act
    if not isinstance(spec, str):
        raise ValueError(f"activation {spec!r} not supported: only named activations can be exported")
    parts = spec.split('-')  # 'name' or '(ad)-name-n', as NES.layers.parse_activation reads it
    name, n = (parts[0], 1.0) if len(parts) == 1 else (parts[1], float(parts[2]) if parts[2] else 1.0)
    if name not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"activation '{name}' not supported, use one of {SUPPORTED_ACTIVATIONS}")
    a = float(to_numpy(layer.a).reshape(())) if layer.adapt else 1.0
    return name, n, a, bool(layer.adapt)


def export(nes, path):
    """Writes the traveltime network of `nes` (NES_OP or NES_TP) to `path`."""
    net = nes.net
    if net.improved_mlp:
        raise ValueError("improved_mlp not supported")
    if net.reciprocity not in (None, 'output'):
        raise ValueError(f"reciprocity {net.reciprocity!r} not supported, only None or 'output'")
    for layer in net.hidden + [net.head]:
        if layer.bias is None or layer.activation.__name__ != 'linear':
            raise ValueError("every Dense layer must have a bias and no activation of its own")

    hidden = [activation_spec(act) for act in net.acts]
    if len({(name, n, adaptive) for name, n, _, adaptive in hidden}) != 1:
        raise ValueError("all hidden layers must use the same activation")
    out_name, out_n, out_a, out_adaptive = activation_spec(net.out_act)

    number = lambda x: repr(float(x))  # shortest form that reads back to the same float64
    numbers = lambda xs: ' '.join(number(x) for x in xs)
    header = {'format': 'nes-weights', 'version': FORMAT_VERSION,
              'solver': 'NES_TP' if net.two_point else 'NES_OP', 'dim': net.dim}
    if net.two_point:
        header['reciprocity'] = net.reciprocity or 'none'
    else:
        header['xs'] = numbers(net.xs.reshape(-1))
    header.update({'input_scale': number(net.scale), 'factored': int(net.factored),
                   'out_vscale': int(net.out_vscale), 'slow_min': number(net.slow_min),
                   'slow_max': number(net.slow_max),
                   'widths': ' '.join(str(n) for n in [net.n_features] + [h.units for h in net.hidden]),
                   'act': hidden[0][0], 'act_n': number(hidden[0][1]), 'act_a': numbers(a for _, _, a, _ in hidden),
                   'act_adaptive': int(hidden[0][3]),
                   'out_act': out_name, 'out_act_n': number(out_n), 'out_act_a': number(out_a),
                   'out_act_adaptive': int(out_adaptive)})
    text = ''.join(f'{key} = {value}\n' for key, value in header.items()) + 'end_header\n'

    arrays = []
    for layer in net.hidden + [net.head]:
        arrays += [layer.kernel, layer.bias]
    # float64, little-endian ('<f8'), C order: a kernel (n_in, n_out) is written row by row
    data = b''.join(np.ascontiguousarray(to_numpy(a), dtype='<f8').tobytes() for a in arrays)
    pathlib.Path(path).write_bytes(text.encode('ascii') + data)


def load(model_dir):
    """Loads a saved NES model; NES_OP or NES_TP is told by the source position in its config."""
    os.environ.setdefault('KERAS_BACKEND', 'jax')
    import NES

    path = pathlib.Path(model_dir)
    with open(path / f'{path.name}_config.pkl', 'rb') as f:
        config = pickle.load(f)
    return (NES.NES_OP if 'xs' in config else NES.NES_TP).load(path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    export(load(sys.argv[1]), sys.argv[2])
    print(f"Wrote {sys.argv[2]}")
