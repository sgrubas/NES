"""Load a weight file (.nesw, docs/WEIGHTS_FORMAT.md) into a NES model: the inverse of export_nes_weights.py.

    from import_nes_weights import load_into
    load_into(nes, "trained.nesw")     # nes: a NES_OP or NES_TP built with build_model

`nes` must have the architecture the file describes (same solver, source, widths, activations ...),
otherwise ValueError names the first difference. Weights then replace the network's values.
"""
import pathlib

import numpy as np

# Header keys that hold one number / a list of numbers / text (all other keys are integers)
NUMBER_KEYS = ('input_scale', 'slow_min', 'slow_max', 'act_n', 'out_act_n', 'out_act_a')
LIST_KEYS = ('xs', 'act_a')


def read_weights(path):
    """Returns `(header, arrays)`: the header as a dict (numbers converted, 'xs', 'act_a' and 'widths' as lists)
    and the weights as a list of float64 arrays in file order: kernel (n_in, n_out), bias (n_out,) for each
    hidden layer, then the head kernel (n_L, 1) and the head bias (1,)."""
    raw = pathlib.Path(path).read_bytes()
    marker = b'end_header\n'
    end = raw.find(marker)
    if end < 0:
        raise ValueError(f"{path}: no 'end_header' line")
    header = {}
    for line in raw[:end].decode('ascii').splitlines():
        key, _, value = line.partition(' = ')
        header[key.strip()] = value.strip()
    if header.get('format') != 'nes-weights' or header.get('version') != '1':
        raise ValueError(f"{path}: not a version 1 'nes-weights' file")

    for key, value in header.items():
        if key in LIST_KEYS:
            header[key] = [float(v) for v in value.split()]
        elif key == 'widths':
            header[key] = [int(v) for v in value.split()]
        elif key in NUMBER_KEYS:
            header[key] = float(value)
        elif key not in ('format', 'solver', 'reciprocity', 'act', 'out_act'):
            header[key] = int(value)

    widths = header['widths']
    shapes = []
    for n_in, n_out in zip(widths[:-1], widths[1:]):
        shapes += [(n_in, n_out), (n_out,)]
    shapes += [(widths[-1], 1), (1,)]                      # head
    sizes = [int(np.prod(s)) for s in shapes]
    data = np.frombuffer(raw, dtype='<f8', offset=end + len(marker))
    if data.size != sum(sizes):
        raise ValueError(f"{path}: expected {sum(sizes)} weights, found {data.size}")
    arrays = [chunk.reshape(s) for chunk, s in zip(np.split(data, np.cumsum(sizes)[:-1]), shapes)]
    return header, arrays


def _check(name, file_value, model_value):
    if not np.array_equal(np.asarray(file_value), np.asarray(model_value)):
        raise ValueError(f"file and model differ in {name}: file {file_value}, model {model_value}")


def _activation_parts(layer):
    """(name, n, adaptive) of an AdaptiveActivation (same reading as the exporter)."""
    parts = layer.act.split('-')
    if len(parts) == 1:
        return parts[0], 1.0, False
    return parts[1], float(parts[2]) if parts[2] else 1.0, 'ad' in parts[0]


def check_matches(header, net):
    """Raises ValueError if the network described by `header` is not the one in `net`."""
    two_point = header['solver'] == 'NES_TP'
    _check('solver', header['solver'], 'NES_TP' if net.two_point else 'NES_OP')
    _check('dim', header['dim'], net.dim)
    if two_point:
        _check('reciprocity', header['reciprocity'], net.reciprocity or 'none')
    else:
        _check('xs', header['xs'], net.xs.reshape(-1))
    _check('input_scale', header['input_scale'], net.scale)
    _check('factored', header['factored'], int(net.factored))
    _check('out_vscale', header['out_vscale'], int(net.out_vscale))
    _check('slow_min', header['slow_min'], net.slow_min)
    _check('slow_max', header['slow_max'], net.slow_max)
    _check('widths', header['widths'], [net.n_features] + [h.units for h in net.hidden])

    for act in net.acts:
        _check('hidden activation', (header['act'], header['act_n'], bool(header['act_adaptive'])),
               _activation_parts(act))
    _check('output activation', (header['out_act'], header['out_act_n'], bool(header['out_act_adaptive'])),
           _activation_parts(net.out_act))


def load_into(nes, path):
    """Assigns the weights in `path` to the variables of `nes.net` (kernels, biases and the trainable `a`)."""
    net = nes.net
    header, arrays = read_weights(path)
    check_matches(header, net)

    for layer, kernel, bias in zip(net.hidden + [net.head], arrays[0::2], arrays[1::2]):
        layer.kernel.assign(kernel)
        layer.bias.assign(bias)
    for act, a in zip(net.acts, header['act_a']):
        if act.adapt:
            act.a.assign([a])
    if net.out_act.adapt:
        net.out_act.a.assign([header['out_act_a']])
