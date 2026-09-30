"""
Generates the legacy fixtures in this folder with the ORIGINAL TensorFlow/Keras-2 NES (commit fac0393).
Not part of the test suite. Run in a separate environment:

    pip install "tensorflow-cpu==2.15.1" "numpy<2" scipy h5py importlib_resources
    git worktree add /tmp/nes-old fac0393
    NES_OLD=/tmp/nes-old python tests/data/make_legacy_fixtures.py
"""
import os
import sys
import pathlib
import numpy as np

sys.path.insert(0, os.environ['NES_OLD'])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf  # noqa: E402
import NES  # noqa: E402
from NES.velocity import VerticalGradient  # noqa: E402

HERE = pathlib.Path(__file__).parent
tf.keras.utils.set_random_seed(0)
rng = np.random.default_rng(1)
vel = VerticalGradient(v0=2.0, a=1.0, xmin=[0., 0.], xmax=[1., 1.])
vel(np.array([[0., 0.], [1., 1.]]))  # old class sets min/max on the first call
refs = {}


def record(tag, nes, keys, n_in):
    x = rng.uniform(0, 1, size=(64, n_in))
    refs[f'{tag}/x'] = x
    for k in keys:
        refs[f'{tag}/{k}'] = nes._predict(x, k, verbose=0)


for recip in [True, False]:
    for imlp in [False, True]:
        tag = f'TP_recip{recip}_imlp{imlp}'
        nes = NES.NES_TP(velocity=vel, name=tag)
        nes.build_model(nl=3, nu=16, reciprocity=recip, improved_mlp=imlp)
        nes.train(x_train=2000, epochs=3, batch_size=500, verbose=0)
        nes.save(str(HERE / 'legacy_models' / tag))
        record(tag, nes, ['T', 'Gr', 'Gs', 'Er', 'Es'], 4)
        refs[f'{tag}/Lr'] = nes.LaplacianR(refs[f'{tag}/x'], verbose=0)
        refs[f'{tag}/Hr'] = nes.HessianR(refs[f'{tag}/x'], verbose=0)
        refs[f'{tag}/Hsr'] = nes.HessianSR(refs[f'{tag}/x'], verbose=0)

for act, out_act, imlp in [('ad-gauss-1', 'ad-sigmoid-1', False), ('tanh', 'sigmoid', False),
                           ('ad-gauss-1', 'ad-sigmoid-1', True)]:
    tag = f'OP_{act}_imlp{imlp}'
    nes = NES.NES_OP(xs=[0.3, 0.2], velocity=vel, name=tag)
    nes.build_model(nl=3, nu=16, act=act, out_act=out_act, improved_mlp=imlp)
    nes.train(x_train=2000, epochs=3, batch_size=500, verbose=0)
    nes.save(str(HERE / 'legacy_models' / tag))
    record(tag, nes, ['T', 'G', 'E'], 2)
    refs[f'{tag}/L'] = nes.Laplacian(refs[f'{tag}/x'], verbose=0)
    refs[f'{tag}/H'] = nes.Hessian(refs[f'{tag}/x'], verbose=0)

np.savez_compressed(HERE / 'legacy_refs.npz', **refs)
print('saved', len(refs), 'arrays')
