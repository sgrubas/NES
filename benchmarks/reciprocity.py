"""
Speed and accuracy of the reciprocity modes of NES-TP on an analytic model (vertical velocity gradient).

    KERAS_BACKEND=jax python benchmarks/reciprocity.py --epochs 300
    KERAS_BACKEND=torch python benchmarks/reciprocity.py --epochs 5 --no-accuracy   # speed only

On Google Colab (GPU runtime), in a cell:
    !git clone -q -b keras3 https://github.com/sgrubas/NES && pip -q install -e NES
    !for b in jax tensorflow torch; do KERAS_BACKEND=$b python NES/benchmarks/reciprocity.py --no-accuracy \
        --epochs 10 --n-train 100000 --batch 25000; done

Prints one JSON line per (mode, seed): parameters, median epoch time (compilation excluded), final loss,
RMAE = mean|T - T_ref| / mean|T_ref| on random source-receiver pairs.
"""
import argparse
import json
import time
import numpy as np
import keras
import NES
from NES.velocity import VerticalGradient, LuneburgLens, MaxwellFishEye


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.t0)


def make_case(name, args, rng):
    """ Velocity model and test pairs (N, 4) with exact traveltimes """
    if name == 'vgrad':      # v = v0 + a z, smooth, no caustics
        vel = VerticalGradient(v0=args.v0, a=args.a, xmin=[0., 0.], xmax=[1., 1.])
        x = rng.uniform(0, 1, (20000, 4))
    elif name == 'luneburg':  # local low-velocity lens in a homogeneous box; exact inside the lens
        vel = LuneburgLens(v_out=2.0, R=1.0, xmin=[-1.2, -1.2], xmax=[1.2, 1.2])
        r, a = np.sqrt(rng.uniform(0, 1, (2, 20000))), rng.uniform(0, 2 * np.pi, (2, 20000))
        x = np.concatenate([np.stack([r[0] * np.cos(a[0]), r[0] * np.sin(a[0])], -1),
                            np.stack([r[1] * np.cos(a[1]), r[1] * np.sin(a[1])], -1)], -1)
    elif name == 'fisheye':  # Maxwell fish-eye, low velocity at the centre, v x3 at the corners
        vel = MaxwellFishEye(v0=1.0, R=1.0, xmin=[-1., -1.], xmax=[1., 1.])
        x = np.concatenate([rng.uniform(-0.6, 0.6, (20000, 2)), rng.uniform(-1, 1, (20000, 2))], -1)
    else:
        raise ValueError(name)
    x = x[np.abs(x[:, 2:] - x[:, :2]).sum(-1) > 1e-2]
    return vel, x, vel.time(x[:, 2:], x[:, :2])


def run(mode, seed, args, vel, x_test, t_test):
    keras.utils.set_random_seed(seed)
    nes = NES.NES_TP(vel)
    nes.build_model(nl=args.nl, nu=args.nu, reciprocity=mode, improved_mlp=args.improved_mlp)
    timer = EpochTimer()
    h = nes.train(args.n_train, epochs=args.epochs, batch_size=args.batch, callbacks=[timer], verbose=0)
    out = dict(backend=keras.backend.backend(), velocity=args.velocity, mode=mode, seed=seed,
               params=nes.net.count_params(),
               epoch_s=float(np.median(timer.times[1:])) if len(timer.times) > 1 else timer.times[0],
               steps_per_epoch=int(np.ceil(len(nes.x_train['x']) / args.batch)),
               loss=float(h.history['loss'][-1]))
    if not args.no_accuracy:
        t = nes.Traveltime(x_test)
        out['rmae_%'] = float(100 * np.abs(t - t_test).mean() / np.abs(t_test).mean())
    print(json.dumps(out), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modes', nargs='+', default=['output', 'first_layer', 'invariant'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--n-train', type=int, default=50000)
    p.add_argument('--batch', type=int, default=12500)
    p.add_argument('--nl', type=int, default=4)
    p.add_argument('--nu', type=int, default=50)
    p.add_argument('--improved-mlp', action='store_true')
    p.add_argument('--no-accuracy', action='store_true')
    p.add_argument('--v0', type=float, default=1.0)
    p.add_argument('--a', type=float, default=3.0)
    p.add_argument('--velocity', default='vgrad', choices=['vgrad', 'luneburg', 'fisheye'])
    args = p.parse_args()

    vel, x_test, t_test = make_case(args.velocity, args, np.random.default_rng(123))
    for seed in args.seeds:
        for mode in args.modes:
            run(mode, seed, args, vel, x_test, t_test)


if __name__ == '__main__':
    main()
