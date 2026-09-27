"""
Ablation of the hard-constraint NES-OP prototypes on the analytic test models.

    KERAS_BACKEND=jax python prototypes/run_ablation.py --out <results dir>

Writes, per model: fmm_<model>.json (FMM error and time per grid), <model>__<config>.json (metrics, training curve)
and <model>__<config>.png (isochrones and error map); progress.json after every run. Prints one `EVENT ...` line
every `--event-every` runs, at the end of each model and at the end.
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import problems as P
import hard_nes as H

C = H.Config
CONFIGS = (
    # hard constraints x optimizer, small network (3 x 16), NES initialization
    [C('nes', 1, 'adam'), C('fermat', 1, 'adam'), C('nes', 3, 'adam'), C('fermat', 3, 'adam'),
     C('nes', 1, 'lm'), C('fermat', 1, 'lm'), C('nes', 3, 'lm'), C('fermat', 3, 'lm')]
    # the same with domain-covering ridge initialization
    + [C(c.factor, c.heads, c.opt, init='ridge') for c in
       [C('nes', 1, 'adam'), C('fermat', 1, 'adam'), C('nes', 3, 'adam'), C('fermat', 3, 'adam'),
        C('nes', 1, 'lm'), C('fermat', 1, 'lm'), C('nes', 3, 'lm'), C('fermat', 3, 'lm')]]
    # vanishing viscosity
    + [C('nes', 1, 'adam', visc=True), C('fermat', 1, 'adam', visc=True)]
    # NES as published (5 x 32), and Fermat at that size
    + [C('nes', 1, 'adam', nl=5, nu=32), C('fermat', 1, 'adam', nl=5, nu=32)]
)
MODEL_ORDER = ['GaussLow', 'TwoGaussLow', 'GaussHigh', 'HyperbolicLens', 'VerticalGradient']
LABEL = {'nes': 'NES factor', 'fermat': 'Fermat factor'}


def label(cfg):
    parts = [LABEL[cfg.factor], 'min of 3 heads' if cfg.heads > 1 else '1 head',
             'Levenberg-Marquardt' if cfg.opt == 'lm' else 'Adam', f'{cfg.init} init']
    if cfg.visc:
        parts.append('vanishing viscosity')
    return ', '.join(parts) + f', {cfg.nl}x{cfg.nu}'


def downsample(curve, n=80):
    """ Keep the first point and points spaced geometrically in time (the charts are log-log) """
    t = np.array([c[1] for c in curve])
    if len(curve) <= n:
        idx = np.arange(len(curve))
    else:
        pos = np.flatnonzero(t > 0)
        grid = np.geomspace(t[pos[0]], t[-1], n - 1)
        idx = np.unique(np.concatenate([[0], np.searchsorted(t, grid).clip(0, len(t) - 1), [len(t) - 1]]))
    return dict(step=[int(curve[i][0]) for i in idx], t=[round(float(curve[i][1]), 4) for i in idx],
                rmae=[float(curve[i][3]) for i in idx])


def figure(path, prob, cfg, m, title=None):
    x, z = prob.axes
    T, Tr = m['T'].reshape(prob.X.shape[:2]), prob.T_ref
    err = np.log10(np.maximum(np.abs(m['rel']).reshape(prob.X.shape[:2]), 1e-9))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.7), constrained_layout=True)
    im = a1.pcolormesh(x, z, prob.vel(prob.X).T, cmap='viridis', shading='auto', rasterized=True)
    levels = np.linspace(np.nanmin(Tr), np.nanmax(Tr), 15)
    a1.contour(x, z, Tr.T, levels, colors='black', linewidths=2.2)
    a1.contour(x, z, T.T, levels, colors='white', linewidths=1.0, linestyles='dashed')
    fig.colorbar(im, ax=a1, shrink=0.85, label='velocity')
    a1.legend([Line2D([], [], color='black', lw=2.2), Line2D([], [], color='white', lw=1.0, ls='--')],
              ['reference', 'NES'], loc='upper right', fontsize=8, frameon=True, facecolor='lightgrey',
              framealpha=0.85)
    im2 = a2.pcolormesh(x, z, err.T, cmap='Blues', vmin=-7, vmax=-2, shading='auto', rasterized=True)
    fig.colorbar(im2, ax=a2, shrink=0.85, label='log10 |relative error|')
    for a, t in [(a1, 'isochrones'), (a2, f'RMAE {100 * m["rmae"]:.2g} %')]:
        a.plot(*prob.xs, marker='*', color='red', ms=11, mec='none')
        a.set(aspect='equal', xlim=(x[0], x[-1]), ylim=(z[-1], z[0]), xlabel='x', title=t)
    a1.set_ylabel('z')
    fig.suptitle(f'{prob.title}: {title or label(cfg)}', fontsize=9)
    fig.savefig(path, dpi=105)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='*', default=MODEL_ORDER)
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--lm-iters', type=int, default=200)
    ap.add_argument('--event-every', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    total = len(args.models) * len(CONFIGS)
    done, t_start = 0, time.time()

    def progress(current):
        with open(os.path.join(args.out, 'progress.json'), 'w') as f:
            json.dump(dict(total=total, done=done, current=current, started=t_start, updated=time.time()), f)

    for mi, name in enumerate(args.models):
        title, vel, xs, exact, caustic = P.MODELS[name]
        progress(f'{name}: reference and FMM benchmark')
        T_ref, kind = P.reference(name)
        with open(os.path.join(args.out, f'fmm_{name}.json'), 'w') as f:
            json.dump(dict(model=name, title=title, caustic=caustic, reference=kind, order=mi,
                           rows=P.fmm_benchmark(name)), f)
        prob = H.Problem(name, title, vel, xs, T_ref, caustic=caustic, reference=kind)
        print(f'EVENT fmm model={name}', flush=True)

        for ci, cfg in enumerate(CONFIGS):
            rid = f'{name}__{cfg.id}'
            progress(f'{title}: {label(cfg)}')
            rec = dict(id=rid, model=name, model_title=title, caustic=caustic, reference=kind,
                       cfg=cfg.as_dict(), label=label(cfg), order=mi * 100 + ci)
            try:
                kw = dict(epochs=args.epochs) if cfg.opt == 'adam' else dict(iters=args.lm_iters)
                p, curve, compile_s = H.train(prob, cfg, **kw)
                m = H.Evaluator(prob, cfg).full(p)
                figure(os.path.join(args.out, f'{rid}.png'), prob, cfg, m)
                rec.update(status='done', params=H.count_params(p), compile_s=round(compile_s, 2),
                           train_s=round(curve[-1][1], 2), steps=int(curve[-1][0]),
                           **{k: m[k] for k in ('rmae', 'max_over', 'max_under', 'cert_over', 'cert_under')},
                           curve=downsample(curve))
            except Exception as e:
                traceback.print_exc()
                rec.update(status='failed', error=f'{type(e).__name__}: {e}')
                print(f'EVENT failed run={rid} error={type(e).__name__}', flush=True)
            rec['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            with open(os.path.join(args.out, f'{rid}.json'), 'w') as f:
                json.dump(rec, f)
            done += 1
            progress('')
            if done % args.event_every == 0 and ci != len(CONFIGS) - 1:
                print(f'EVENT progress done={done} total={total} model={name}', flush=True)
        print(f'EVENT model_done model={name} done={done} total={total}', flush=True)
    progress('finished')
    print(f'EVENT all_done total={total} minutes={(time.time() - t_start) / 60:.1f}', flush=True)


if __name__ == '__main__':
    main()
