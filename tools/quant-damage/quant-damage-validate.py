#!/usr/bin/env python3
# Compares sampled llama-quant-damage estimates with the exact values in the same JSON (run with --exact --repeat N).

import argparse
import json
import math
import statistics
import sys


def kendall_tau(a, b):
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            x = (a[i] - a[j]) * (b[i] - b[j])
            if x > 0:
                conc += 1
            elif x < 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else 1.0


def pct(v, p):
    v = sorted(v)
    if not v:
        return float('nan')
    k = min(len(v) - 1, max(0, int(math.ceil(p / 100.0 * len(v))) - 1))
    return v[k]


def fmt(x, f='{:.3g}'):
    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) else f.format(x)


def analyze_tensor(t, key, z, exact_runs=None):
    runs = t.get('runs', [])
    exact = [r for r in runs if r['estimator'] == 'exact'] or (exact_runs or {}).get(t['name'], [])
    sampled = [r for r in runs if r['estimator'] != 'exact']
    if not exact or not sampled:
        return None
    ex = exact[0]
    # index everything by the sampled runs' type order; the exact run may cover more types
    names = [x['qtype'] for x in sampled[0]['types']]
    ex_by_name = {x['qtype']: x for x in ex['types']}
    ok = [i for i, q in enumerate(names) if q in ex_by_name and ex_by_name[q]['status'] == 'ok' and ex_by_name[q].get(key) is not None
          and ex_by_name[q]['R'] > 1e-9 and sampled[0]['types'][i]['status'] == 'ok']
    exv = {i: ex_by_name[names[i]][key] for i in ok}

    per_type = {}
    for i in ok:
        errs, cover, rse = [], 0, []
        for r in sampled:
            x = r['types'][i]
            v, se = x[key], x[key + '_se']
            if v is None:
                continue
            errs.append((v - exv[i]) / exv[i])
            if se is not None and abs(v - exv[i]) <= z * se:
                cover += 1
            if se is not None and v > 0:
                rse.append(se / v)
        per_type[i] = {
            'bias': statistics.mean(errs) if errs else float('nan'),
            'abs_med': statistics.median([abs(e) for e in errs]) if errs else float('nan'),
            'abs_p95': pct([abs(e) for e in errs], 95),
            'cover': cover / len(sampled),
            'rse': statistics.median(rse) if rse else float('nan'),
        }

    # final-round ranking quality
    taus, exact_order, wrong_confident, wrong_pairs = [], 0, 0, 0
    ex_list = [exv[i] for i in ok]
    for r in sampled:
        vals = [r['types'][i][key] for i in ok]
        ses = [r['types'][i][key + '_se'] or 0.0 for i in ok]
        taus.append(kendall_tau(ex_list, vals))
        if sorted(range(len(ok)), key=lambda k: vals[k]) == sorted(range(len(ok)), key=lambda k: ex_list[k]):
            exact_order += 1
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                if (ex_list[a] - ex_list[b]) * (vals[a] - vals[b]) < 0:
                    wrong_pairs += 1
                    if abs(vals[a] - vals[b]) > 2.0 * math.sqrt(ses[a] ** 2 + ses[b] ** 2):
                        wrong_confident += 1

    # convergence by round (R only; rounds store the primary R)
    conv = []
    if key == 'R':
        n_rounds = min(len(r['rounds']) for r in sampled)
        for k in range(n_rounds):
            n_units = statistics.median([r['rounds'][k]['n_units'] for r in sampled])
            errs, cover, tau, maxrse = [], 0, [], []
            tot = 0
            for r in sampled:
                rd = r['rounds'][k]
                vals = [rd['R'][i] for i in ok]
                ses = [rd['R_se'][i] for i in ok]
                for i, v, se in zip(ok, vals, ses):
                    if v is None:
                        continue
                    errs.append(abs(v - exv[i]) / exv[i])
                    tot += 1
                    if se is not None and abs(v - exv[i]) <= z * se:
                        cover += 1
                tau.append(kendall_tau(ex_list, vals))
                maxrse.append(max((se / v) for v, se in zip(vals, ses) if v and se is not None))
            conv.append({
                'n': n_units, 'frac': n_units / sampled[0]['units_total'],
                'err_med': statistics.median(errs), 'err_p95': pct(errs, 95),
                'cover': cover / tot if tot else float('nan'),
                'tau_min': min(tau), 'tau_mean': statistics.mean(tau), 'rse_max': statistics.median(maxrse),
            })

    return {
        'names': names, 'ok': ok, 'exv': exv, 'per_type': per_type, 'n_seeds': len(sampled),
        'units': statistics.median([r['units_sampled'] for r in sampled]), 'units_total': sampled[0]['units_total'],
        'tau_mean': statistics.mean(taus), 'tau_min': min(taus), 'exact_order': exact_order,
        'wrong_pairs': wrong_pairs, 'wrong_confident': wrong_confident, 'conv': conv,
        'estimator': sampled[0]['estimator'],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('json', nargs='+')
    ap.add_argument('--key', default='R', help='R, D or rel_mse')
    ap.add_argument('--z', type=float, default=1.96)
    ap.add_argument('--types', action='store_true', help='per-type tables')
    ap.add_argument('--exact-from', help='JSON with the exact runs (--exact --no-sampled)')
    args = ap.parse_args()

    exact_runs = {}
    if args.exact_from:
        with open(args.exact_from) as f:
            for t in json.load(f)['tensors']:
                exact_runs[t['name']] = [r for r in t.get('runs', []) if r['estimator'] == 'exact']

    summary = []
    for path in args.json:
        with open(path) as f:
            data = json.load(f)
        for t in data['tensors']:
            a = analyze_tensor(t, args.key, args.z, exact_runs)
            if a is None:
                continue
            summary.append((t['name'], a))
            if args.types:
                print(f"\n### {t['name']} {t['shape']} {t['source_dtype']} ({a['estimator']}, {a['n_seeds']} seeds, median {a['units']:.0f}/{a['units_total']} rows)\n")
                print(f"| qtype | exact {args.key} | bias | median abs err | p95 abs err | median rse | CI coverage |")
                print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
                for i in sorted(a['ok'], key=lambda i: a['exv'][i]):
                    p = a['per_type'][i]
                    print(f"| {a['names'][i]} | {a['exv'][i]:.4e} | {p['bias']*100:+.2f}% | {p['abs_med']*100:.2f}% | {p['abs_p95']*100:.2f}% | {p['rse']*100:.2f}% | {p['cover']*100:.0f}% |")
                if a['conv']:
                    print(f"\n| rows | fraction | median abs err | p95 abs err | CI coverage | Kendall tau mean | tau min | median max rse |")
                    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                    for c in a['conv']:
                        print(f"| {c['n']:.0f} | {c['frac']*100:.2f}% | {c['err_med']*100:.2f}% | {c['err_p95']*100:.2f}% | {c['cover']*100:.0f}% | {c['tau_mean']:.4f} | {c['tau_min']:.4f} | {c['rse_max']*100:.2f}% |")

    print(f"\n| tensor | rows | seeds | types | median abs err | worst p95 abs err | CI coverage | tau mean | tau min | exact order | wrong pairs (confident) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, a in summary:
        pt = a['per_type'].values()
        med = statistics.median([p['abs_med'] for p in pt])
        worst = max(p['abs_p95'] for p in pt)
        cov = statistics.mean([p['cover'] for p in pt])
        print(f"| {name} | {a['units']:.0f}/{a['units_total']} | {a['n_seeds']} | {len(a['ok'])} | {med*100:.2f}% | {worst*100:.2f}% | {cov*100:.0f}% | {a['tau_mean']:.4f} | {a['tau_min']:.4f} | {a['exact_order']}/{a['n_seeds']} | {a['wrong_pairs']} ({a['wrong_confident']}) |")


if __name__ == '__main__':
    sys.exit(main())
