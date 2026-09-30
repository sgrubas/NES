"""
Loading of models saved by the original TensorFlow / Keras-2 NES (versions <= 0.2.x).

Old weights are mapped onto `TraveltimeNet.legacy_variables()`, which lists the variables in the order
of the original Keras-2 weight lists. Old models used output-averaged reciprocity ('output').
"""
import re
import warnings
import numpy as np
import keras

from .eikonalLayers import IsoEikonal

_NON_BUILD_KEYS = ('velocity', 'xs', 'name', 'equation.config')


def _legacy_activation(act):
    """ Old versions silently used n = 1 when 'n' was not an integer string (e.g. 'ad-gauss-0.5') """
    if isinstance(act, str) and act.count('-') == 2:
        ad, name, n = act.split('-')
        if not n.isdigit():
            return f'{ad}-{name}-1'
    return act


def _read_weights(path):
    filename = lambda kw: path / f'{path.name}_{kw}'
    if filename('weights.pkl').is_file():
        import pickle
        with open(filename('weights.pkl'), 'rb') as f:
            return pickle.load(f)
    import h5py
    with h5py.File(filename('weights.h5'), 'r') as f:
        weights = []
        for layer in f.attrs['layer_names']:
            layer = layer.decode() if isinstance(layer, bytes) else layer
            if layer.endswith('SourceLoc'):  # source location of NES-OP, not a trainable weight
                continue
            group = f[layer]
            weights += [np.array(group[w]) for w in group.attrs['weight_names']]
    return weights


def _assign(variables, weights, path):
    if len(variables) != len(weights) or any(tuple(v.shape) != np.shape(w) for v, w in zip(variables, weights)):
        raise ValueError(f"Weights in '{path}' do not match the architecture: "
                         f"{[tuple(v.shape) for v in variables]} vs {[np.shape(w) for w in weights]}")
    for v, w in zip(variables, weights):
        v.assign(w)


def _restore_optimizer(instance, state):
    config, weights = state['optimizer.config'], state['optimizer.weights']
    if config.get('name', '').lower() != 'adam':
        warnings.warn(f"Optimizer '{config.get('name')}' of the old model is not restored, compile() anew")
        return
    lr, decay = config['learning_rate'], config.get('decay', 0.0)
    schedule = keras.optimizers.schedules.InverseTimeDecay(lr, 1, decay) if decay else lr
    optimizer = keras.optimizers.Adam(learning_rate=schedule, beta_1=config['beta_1'], beta_2=config['beta_2'],
                                      epsilon=config['epsilon'], amsgrad=config.get('amsgrad', False),
                                      weight_decay=config.get('weight_decay'))
    optimizer.build(instance.model.trainable_variables)
    variables = instance.net.legacy_variables()
    n = len(variables)
    try:  # Keras-2 layout: [iterations, m_1..m_n, v_1..v_n]
        if len(weights) < 1 + 2 * n:
            raise ValueError('unexpected number of optimizer weights')
        optimizer.iterations.assign(weights[0])
        for k, var in enumerate(variables):
            i = optimizer._get_variable_index(var)
            optimizer._momentums[i].assign(weights[1 + k])
            optimizer._velocities[i].assign(weights[1 + n + k])
    except Exception as e:  # moments are a convenience, hyperparameters and step count are what matters
        warnings.warn(f"Adam moments of the old model were not restored ({e})")
    instance.compile(optimizer=optimizer, loss=state.get('loss', 'mae'))


def _convert_x_train(x_train, instance):
    """ {'<name>_xs0': ..., '<name>_xr1': ..., '<name>_vr': ...} -> {'x': (N, n_in), 'vr': (N, 1)} """
    columns, out = {}, {}
    for key, value in x_train.items():
        coord = re.search(r'_(xs|xr)(\d+)$', key)
        vel = re.search(r'_(v|vr|vs)$', key)
        if coord:
            offset = 0 if coord.group(1) == 'xs' else instance._n_in - instance.dim
            columns[offset + int(coord.group(2))] = np.asarray(value).reshape(-1)
        elif vel:
            out[vel.group(1)] = np.asarray(value, dtype='float32').reshape(-1, 1)
    out['x'] = np.stack([columns[i] for i in sorted(columns)], axis=-1).astype('float32')
    return out


def load(cls, path, config):
    import pickle
    filename = lambda kw: path / f'{path.name}_{kw}'

    build = {k: v for k, v in config.items() if k not in _NON_BUILD_KEYS}
    for key in ('act', 'out_act'):
        if key in build:
            build[key] = _legacy_activation(build[key])
    if 'reciprocity' in build:
        build['reciprocity'] = 'output' if build['reciprocity'] else None

    init = dict(velocity=config['velocity'], name=config.get('name'),
                eikonal=IsoEikonal.from_config(config['equation.config']))
    if 'xs' in config:
        init['xs'] = config['xs']
    instance = cls(**init)
    instance.build_model(**build)
    _assign(instance.net.legacy_variables(), _read_weights(path), path)
    print(f'Loaded model from "{path}" (saved by NES <= 0.2, TensorFlow/Keras 2)')

    if filename('optimizer.pkl').is_file():
        with open(filename('optimizer.pkl'), 'rb') as f:
            state = pickle.load(f)
        if state is not None:
            _restore_optimizer(instance, state)
            print('Compiled the model with saved optimizer')
    if filename('train_data.pkl').is_file():
        with open(filename('train_data.pkl'), 'rb') as f:
            instance.x_train = _convert_x_train(pickle.load(f), instance)
        print(f'Loaded last training data: see {cls.__name__}.x_train')
    return instance
