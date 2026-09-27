"""Summary of the ablation: best per model, main effects of each component, FMM grid matching the best error."""
import glob
import json
import os
import sys
import numpy as np

res = sys.argv[1]
runs = [json.load(open(f)) for f in glob.glob(os.path.join(res, '*__*.json'))]
runs = [r for r in runs if r.get('status') == 'done']
fmm = {json.load(open(f))['model']: json.load(open(f)) for f in glob.glob(os.path.join(res, 'fmm_*.json'))}
models = [m for m in ['GaussLow', 'TwoGaussLow', 'GaussHigh', 'HyperbolicLens', 'VerticalGradient'] if m in fmm]
key = lambda c: (c['factor'], c['heads'], c['opt'], c['init'], bool(c['visc']), c['nl'], c['nu'])
by = {(r['model'], key(r['cfg'])): r for r in runs}
BASE = ('nes', 1, 'adam', 'he', False, 5, 32)


def fmm_match(rows, r):
    n = np.log([x['n'] for x in rows]); e = np.log([x['rmae'] for x in rows])
    y = np.log(r)
    if y >= e[0]:
        return rows[0]['n']
    for i in range(len(n) - 1):
        if e[i + 1] <= y <= e[i]:
            return float(np.exp(n[i] + (y - e[i]) * (n[i + 1] - n[i]) / (e[i + 1] - e[i])))
    return float(np.exp(n[-1] + (y - e[-1]) * (n[-1] - n[-2]) / (e[-1] - e[-2])))


print('BEST PER MODEL')
summary = []
for m in models:
    rs = sorted([r for r in runs if r['model'] == m], key=lambda r: r['rmae'])
    b, base = rs[0], by.get((m, BASE))
    if base is None:
        continue
    f241 = next(x for x in fmm[m]['rows'] if x['n'] == 241)
    match = fmm_match(fmm[m]['rows'], b['rmae'])
    row = dict(model=m, title=fmm[m]['title'], best=b['cfg']['id'], best_rmae=b['rmae'], params=b['params'],
               t=b['train_s'], cert=max(b['cert_over'], b['cert_under']), base_rmae=base['rmae'] if base else None,
               gain=base['rmae'] / b['rmae'] if base else None, fmm241=f241['rmae'], fmm241_s=f241['seconds'],
               match_n=match)
    summary.append(row)
    print(f"{m:17s} best {b['cfg']['id']:26s} {100 * b['rmae']:.2e}% ({b['params']} p, {b['train_s']:.0f}s, "
          f"max res {row['cert']:.1e}) | NES 5x32 {100 * row['base_rmae']:.2e}% -> x{row['gain']:.0f} | "
          f"FMM241 {100 * f241['rmae']:.1e}% | FMM grid for same error ~{match:.0f}^2")

COMPONENTS = [('Fermat factor', lambda c: c['factor'] == 'fermat', lambda k: ('nes',) + k[1:]),
              ('Min of 3 heads', lambda c: c['heads'] == 3, lambda k: k[:1] + (1,) + k[2:]),
              ('Levenberg-Marquardt', lambda c: c['opt'] == 'lm', lambda k: k[:2] + ('adam',) + k[3:]),
              ('Ridge init', lambda c: c['init'] == 'ridge', lambda k: k[:3] + ('he',) + k[4:]),
              ('Vanishing viscosity', lambda c: c['visc'], lambda k: k[:4] + (False,) + k[5:]),
              ('5x32 instead of 3x16', lambda c: c['nl'] == 5, lambda k: k[:5] + (3, 16))]
print('\nMAIN EFFECTS (geometric mean of error ratio on/off; <1 = lower error)')
effects = []
for name, on, off in COMPONENTS:
    per, allr = {}, []
    for m in models:
        rr = [by[(m, key(r['cfg']))]['rmae'] / by[(m, off(key(r['cfg'])))]['rmae']
              for r in runs if r['model'] == m and on(r['cfg']) and (m, off(key(r['cfg']))) in by]
        per[m] = float(np.exp(np.mean(np.log(rr)))) if rr else None
        allr += rr
    g = float(np.exp(np.mean(np.log(allr))))
    wins = float(np.mean(np.array(allr) < 1))
    effects.append(dict(name=name, overall=g, per=per, pairs=len(allr), wins=wins))
    print(f"{name:22s} overall x{g:.2f} ({len(allr)} pairs, lower in {100 * wins:.0f}%) | " +
          ' '.join(f"{m[:10]}:{v:.2f}" if v else f'{m[:10]}:-' for m, v in per.items()))
json.dump(dict(summary=summary, effects=effects), open(os.path.join(res, 'analysis.json'), 'w'), indent=1)
