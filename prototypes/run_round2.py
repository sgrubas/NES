"""
Round 2: second-order and quasi-Newton training of the Fermat factorization.

    KERAS_BACKEND=jax python prototypes/run_round2.py --out <results dir>

Main grid: every model x factor {nes, fermat-exp, fermat-sq} x optimizer {adam, lm, lbfgsb, lbfgsb after 1000 Adam
epochs}, two seeds (one for Adam), network 3 x 16, ridge initialization, 60 s budget for lm and quasi-Newton.
Long runs: the two models with exact references, fermat-exp, lbfgsb and ssbfgs for 300 s (how far do quasi-Newton
methods go toward machine precision?).
Writes <model>__<run>.json / .png like run_ablation.py, progress.json, and prints EVENT lines.
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
from run_ablation import downsample, figure

MODEL_ORDER = ['GaussLow', 'TwoGaussLow', 'GaussHigh', 'HyperbolicLens', 'VerticalGradient']
FACTORS = ['nes', 'fermat-exp', 'fermat-sq']
LABEL = {'nes': 'NES factor', 'fermat-exp': 'Fermat, log-deficit', 'fermat-sq': 'Fermat, squared deficit',
         'adam': 'Adam', 'lm': 'Levenberg-Marquardt', 'lbfgsb': 'L-BFGS-B', 'ssbfgs': 'self-scaled BFGS'}


def runs_for(model, budget, long_budget):
    specs = []
    for f in FACTORS:
        specs.append(dict(factor=f, opt='adam', seed=0, warmup=0, budget=None))
    for seed in (0, 1):
        for f in FACTORS:
            specs.append(dict(factor=f, opt='lm', seed=seed, warmup=0, budget=budget))
            specs.append(dict(factor=f, opt='lbfgsb', seed=seed, warmup=0, budget=budget))
            specs.append(dict(factor=f, opt='lbfgsb', seed=seed, warmup=1000, budget=budget))
    if P.MODELS[model][3]:                                            # exact reference
        for o in ('lbfgsb', 'ssbfgs'):
            specs.append(dict(factor='fermat-exp', opt=o, seed=0, warmup=0, budget=long_budget, long=True))
    return specs


def run_id(s):
    return (f"{s['factor']}-{s['opt']}" + (f"-w{s['warmup']}" if s['warmup'] else '') +
            ('-long' if s.get('long') else '') + f"-s{s['seed']}")


def label(s):
    opt = LABEL[s['opt']] + (f" after {s['warmup']} Adam epochs" if s['warmup'] else '')
    return f"{LABEL[s['factor']]}, {opt}" + (f", {s['budget']} s" if s.get('long') else '') + f", seed {s['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='*', default=MODEL_ORDER)
    ap.add_argument('--budget', type=float, default=60)
    ap.add_argument('--long-budget', type=float, default=300)
    ap.add_argument('--event-every', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plan = [(m, s) for m in args.models for s in runs_for(m, args.budget, args.long_budget)]
    total, done, t_start = len(plan), 0, time.time()

    def progress(current):
        with open(os.path.join(args.out, 'progress.json'), 'w') as f:
            json.dump(dict(total=total, done=done, current=current, started=t_start, updated=time.time()), f)

    probs = {}
    for mi, name in enumerate(args.models):
        title, vel, xs, exact, caustic = P.MODELS[name]
        progress(f'{name}: reference and FMM benchmark')
        T_ref, kind = P.reference(name)
        with open(os.path.join(args.out, f'fmm_{name}.json'), 'w') as f:
            json.dump(dict(model=name, title=title, caustic=caustic, reference=kind, order=mi,
                           rows=P.fmm_benchmark(name)), f)
        prob = H.Problem(name, title, vel, xs, T_ref, caustic=caustic, reference=kind)
        print(f'EVENT fmm model={name}', flush=True)
        specs = [s for m, s in plan if m == name]
        for ci, s in enumerate(specs):
            cfg = H.Config(s['factor'], 1, s['opt'], init='ridge')
            rid = f'{name}__{run_id(s)}'
            progress(f'{title}: {label(s)}')
            rec = dict(id=rid, model=name, model_title=title, caustic=caustic, reference=kind,
                       cfg=dict(cfg.as_dict(), seed=s['seed'], warmup=s['warmup'], long=bool(s.get('long')),
                                budget=s['budget'], run=run_id(s)), label=label(s), order=mi * 100 + ci)
            try:
                if s['opt'] == 'adam':
                    kw = dict(epochs=3000)
                elif s['opt'] == 'lm':
                    kw = dict(iters=10000, max_seconds=s['budget'], warmup=s['warmup'])
                else:
                    kw = dict(max_seconds=s['budget'], warmup=s['warmup'])
                p, curve, compile_s = H.train(prob, cfg, seed=s['seed'], **kw)
                m = H.Evaluator(prob, cfg).full(p)
                figure(os.path.join(args.out, f'{rid}.png'), prob, cfg, m, title=label(s))
                rec.update(status='done', params=H.count_params(p), compile_s=round(compile_s, 2),
                           train_s=round(curve[-1][1], 2), steps=int(curve[-1][0]),
                           **{k: m[k] for k in ('rmae', 'max_over', 'max_under', 'cert_over', 'cert_under')},
                           curve=downsample(curve, 100))
            except Exception as e:
                traceback.print_exc()
                rec.update(status='failed', error=f'{type(e).__name__}: {e}')
                print(f'EVENT failed run={rid} error={type(e).__name__}', flush=True)
            rec['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            with open(os.path.join(args.out, f'{rid}.json'), 'w') as f:
                json.dump(rec, f)
            done += 1
            progress('')
            if done % args.event_every == 0 and ci != len(specs) - 1:
                print(f'EVENT progress done={done} total={total} model={name}', flush=True)
        print(f'EVENT model_done model={name} done={done} total={total}', flush=True)
    progress('finished')
    print(f'EVENT all_done total={total} minutes={(time.time() - t_start) / 60:.1f}', flush=True)


if __name__ == '__main__':
    main()
