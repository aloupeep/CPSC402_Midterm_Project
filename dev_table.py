#!/usr/bin/env python3
"""Full development tuning table (all configurations, not only winners)."""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path
OUT = Path('/home/worker/lab/campaign/writeup_results'); cases = {json.loads(l)['case_id']: json.loads(l) for l in open(OUT / 'cases.jsonl')}
g = defaultdict(lambda: dict(t=[], f=[], outer=[], inner=[], pass_=0, n=0, term=set()))
for l in open(OUT / 'trials.jsonl'):
    t = json.loads(l)
    if t['phase'] != 'development' or t['run_kind'] == 'warmup': continue
    c = cases[t['case_id']]; fl = t.get('dev_flags') or {}; d = g[(c['n'], c['kappa_nominal'], c['seed'], t['method_id'], t['tag'], fl.get('nb'), fl.get('gmres-m'), fl.get('gmres-tol'), t['factor_diagnostics'].get('panel'))]
    d['t'].append(t['timings_s']['device_ready_wall']); d['f'].append(t['timings_s']['factor_total']); d['outer'].append(t['counters']['outer']); d['inner'].append(t['counters']['inner_total']); d['n'] += 1; d['pass_'] += t['validation']['both_backward'] == 'pass'; d['term'].add(t['terminal_reason'])
with open(OUT / 'summaries' / 'dev_tuning.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['n', 'kappa_nominal', 'seed', 'method_id', 'tag', 'nb', 'gmres_m', 'gmres_tol', 'panel', 'reps', 'device_ready_median_s', 'factor_median_s', 'median_outer', 'median_inner', 'qualified', 'terminal_reasons'])
    for k, d in sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        w.writerow(list(k) + [d['n'], st.median([x for x in d['t'] if x is not None]) if any(x is not None for x in d['t']) else None, st.median([x for x in d['f'] if x is not None]) if any(x is not None for x in d['f']) else None, st.median(d['outer']), st.median(d['inner']), d['pass_'], ';'.join(sorted(x for x in d['term'] if x))])
print('dev_tuning.csv rows', len(g))
