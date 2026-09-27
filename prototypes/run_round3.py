"""
Round 3.

A. Derivatives: accuracy of T, grad T, lap T and hess T of NES (autodiff) against FMM (finite differences), on the
   models with closed forms, against degrees of freedom (network parameters, grid values).
       NES: factors {nes, fermat-exp} x networks {2x8, 3x12, 3x16, 4x24, 5x32}, Levenberg-Marquardt,
            max(4000, 2 x parameters) collocation points, budgets 90 ... 900 s (150+ iterations for every size).
       FMM: grids 41^2 ... 961^2 (problems.FMM_SIZES), 2nd-order central differences.
B. Collocation sampling on the caustic models and one smooth control: Fermat squared deficit, Levenberg-Marquardt
   90 s, collocation set replaced every 40 iterations by: none (fixed), uniform, rad, rad-curv, dwr (hard_nes.Resampler).

    KERAS_BACKEND=jax python prototypes/run_round3.py --out <results dir> [--parts A B]

Writes deriv_fmm_<model>.json, <model>__deriv-<factor>-<nl>x<nu>.json, <model>__samp-<kind>-s<seed>.json (+ .png for
B), progress.json, and prints EVENT lines.
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, os.path.dirname(__file__))
import problems as P
import hard_nes as H
import derivatives as D
from run_ablation import downsample, figure

SIZES = [(2, 8), (3, 12), (3, 16), (4, 24), (5, 32)]
BUDGET = {(2, 8): 90, (3, 12): 120, (3, 16): 150, (4, 24): 300, (5, 32): 900}     # s; 0.03 ... 5.4 s per iteration
N_TRAIN = lambda n_params: max(4000, 2 * n_params)
DERIV_MODELS = ['HyperbolicLens', 'VerticalGradient']
DERIV_FACTORS = ['nes', 'fermat-exp']
SAMP_MODELS = [('GaussLow', 2), ('TwoGaussLow', 2), ('GaussHigh', 1)]
SAMPLERS = ['none', 'uniform', 'rad', 'rad-curv', 'dwr']
SAMP_LABEL = {'none': 'fixed collocation points', 'uniform': 'fresh uniform points', 'rad': 'residual-adaptive',
              'rad-curv': 'residual-adaptive, caustics masked', 'dwr': 'ray-weighted residual'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--parts', nargs='*', default=['A', 'B'])
    ap.add_argument('--event-every', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plan = []
    if 'B' in args.parts:
        plan += [('B', m, dict(kind=k, seed=s)) for m, ns in SAMP_MODELS for s in range(ns) for k in SAMPLERS]
    if 'A' in args.parts:
        plan += [('A', m, dict(factor=f, nl=nl, nu=nu)) for m in DERIV_MODELS for f in DERIV_FACTORS for nl, nu in SIZES]
    total, done, t_start = len(plan), 0, time.time()

    def progress(current):
        with open(os.path.join(args.out, 'progress.json'), 'w') as f:
            json.dump(dict(total=total, done=done, current=current, started=t_start, updated=time.time()), f)

    probs = {}

    def problem(name, n_train=4000):
        if (name, n_train) in probs:
            return probs[name, n_train]
        title, vel, xs, exact, caustic = P.MODELS[name]
        T_ref, kind = P.reference(name)
        seen = any(k[0] == name for k in probs)
        probs[name, n_train] = H.Problem(name, title, vel, xs, T_ref, n_train=n_train, caustic=caustic, reference=kind)
        if not seen:
            if 'A' in args.parts and name in DERIV_MODELS:
                progress(f'{title}: FMM derivatives')
                rows = [D.fmm_derivatives(name, n) for n in P.FMM_SIZES]
                with open(os.path.join(args.out, f'deriv_fmm_{name}.json'), 'w') as f:
                    json.dump(dict(model=name, title=title, rows=rows), f)
                print(f'EVENT deriv_fmm model={name}', flush=True)
            with open(os.path.join(args.out, f'fmm_{name}.json'), 'w') as f:
                json.dump(dict(model=name, title=title, caustic=caustic, reference=kind,
                               order=list(P.MODELS).index(name), rows=P.fmm_benchmark(name)), f)
        return probs[name, n_train]

    last_part = None
    for part, name, spec in plan:
        if part == 'A':
            cfg = H.Config(spec['factor'], 1, 'lm', init='ridge', nl=spec['nl'], nu=spec['nu'])
            n_params = H.count_params(H.init_params(H.jax.random.PRNGKey(0), cfg, problem(name)))
            prob = problem(name, N_TRAIN(n_params))
            rid = f"{name}__deriv-{spec['factor']}-{spec['nl']}x{spec['nu']}"
            label = f"{spec['factor']} factor, {spec['nl']}x{spec['nu']}, Levenberg-Marquardt"
        else:
            prob = problem(name)
            cfg = H.Config('fermat-sq', 1, 'lm', init='ridge')
            rid = f"{name}__samp-{spec['kind']}-s{spec['seed']}"
            label = f"Fermat squared deficit, Levenberg-Marquardt, {SAMP_LABEL[spec['kind']]}, seed {spec['seed']}"
        progress(f'{prob.title}: {label}')
        rec = dict(id=rid, part=part, model=name, model_title=prob.title, caustic=prob.caustic,
                   reference=prob.reference, spec=spec, cfg=cfg.as_dict(), label=label, order=done,
                   n_train=int(prob.train['x'].shape[0]))
        try:
            if part == 'A':
                budget = BUDGET[spec['nl'], spec['nu']]
                p, curve, cs = H.train(prob, cfg, seed=0, iters=100000, max_seconds=budget)
                rec.update(budget=budget, deriv=D.nes_derivatives(prob, cfg, p))
            else:
                sampler = None if spec['kind'] == 'none' else H.Resampler(prob, cfg, spec['kind'], seed=spec['seed'])
                p, curve, cs = H.train(prob, cfg, seed=spec['seed'], iters=100000, max_seconds=90,
                                       sampler=sampler, resample_every=40)
            m = H.Evaluator(prob, cfg).full(p)
            if part == 'B':
                figure(os.path.join(args.out, f'{rid}.png'), prob, cfg, m, title=label)
            rec.update(status='done', params=H.count_params(p), compile_s=round(cs, 2), train_s=round(curve[-1][1], 2),
                       steps=int(curve[-1][0]), **{k: m[k] for k in ('rmae', 'max_over', 'max_under', 'cert_over',
                                                                     'cert_under')}, curve=downsample(curve, 80))
        except Exception as e:
            traceback.print_exc()
            rec.update(status='failed', error=f'{type(e).__name__}: {e}')
            print(f'EVENT failed run={rid} error={type(e).__name__}', flush=True)
        rec['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        with open(os.path.join(args.out, f'{rid}.json'), 'w') as f:
            json.dump(rec, f)
        done += 1
        progress('')
        if done % args.event_every == 0 or part != last_part:
            print(f'EVENT progress done={done} total={total} part={part} model={name}', flush=True)
        last_part = part
    progress('finished')
    print(f'EVENT all_done total={total} minutes={(time.time() - t_start) / 60:.1f}', flush=True)


if __name__ == '__main__':
    main()
