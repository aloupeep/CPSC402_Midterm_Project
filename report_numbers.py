#!/usr/bin/env python3
"""Print the factual numbers used in RUN_SUMMARY.md from the collected contract files."""
import csv, json, statistics as st
from collections import defaultdict, Counter
from pathlib import Path
OUT = Path('/home/worker/lab/campaign/writeup_results'); S = OUT / 'summaries'
cases = {json.loads(l)['case_id']: json.loads(l) for l in open(OUT / 'cases.jsonl')}
trials = [json.loads(l) for l in open(OUT / 'trials.jsonl')]
print('cases', len(cases), 'trials', len(trials), 'non-warmup', sum(t['run_kind'] != 'warmup' for t in trials))
print('by suite', Counter(t['suite'] for t in trials if t['run_kind'] != 'warmup'))
print('by run_kind', Counter(t['run_kind'] for t in trials))
print('terminal reasons', Counter(t['terminal_reason'] for t in trials if t['run_kind'] != 'warmup').most_common(25))
print('runtime errors', Counter((t['method_id'], (t['error'] or '')[:60]) for t in trials if not t['ok']).most_common(10))
print('skipped', Counter((cases[t['case_id']]['n'], t['method_id']) for t in trials if t['skipped']).most_common(10))
# largest n per method with a qualified (primary) timed record
best = defaultdict(lambda: [0, 0])
for t in trials:
    if t['run_kind'] not in ('timed_resident', 'timed_host') or t['suite'] != 'E4': continue
    n = cases[t['case_id']]['n']; key = (t['target_precision'], t['method_id'])
    best[key][0] = max(best[key][0], n)
    if t['validation']['both_backward'] == 'pass': best[key][1] = max(best[key][1], n)
print('largest n run / qualified per (target, method):')
for k in sorted(best): print('  ', k, best[k])
# speedup summary
print('speedup summary (nrhs=1):')
for r in csv.DictReader(open(S / 'e4_speedup_summary.csv')):
    if r['nrhs'] == '1': print('  ', r['target'], r['n'], r['method_id'], (r['method_qualified_cases'] or '0') + '/' + r['total_cases'], 'vs_lu', r['median_ratio_vs_ordinary_lu'], 'vs_refined', r['median_ratio_vs_refined_baseline'])
print('E6 policies (evaluation):')
for r in csv.DictReader(open(S / 'e6_policies.csv')):
    if r['phase'] == 'evaluation': print('  ', r['n'], r['case_family'], r['method_id'], r['pass_primary'] + '/' + r['cases'], r['median_device_ready_s'], r['final_stages'])
print('E3:')
for r in csv.DictReader(open(S / 'e3_scaling.csv')): print('  ', r['n'], r['global_exp'], r['rowcol_span'], r['method_id'], r['pass_primary'] + '/' + r['seeds'], 'nonfinite', r['nonfinite'])
print('E5 reuse:')
for r in csv.DictReader(open(S / 'e5_reuse.csv')): print('  ', r['n'], r['nrhs'], r['method_id'], 'reuse', r['reuse_solve_median_s'], 'fresh', r['fresh_solve_median_s'], 'factor', r['factor_median_s'])
print('E4 timing medians nrhs=1 geometric (device-ready):')
for r in csv.DictReader(open(S / 'e4_timing_per_case_method.csv')):
    if r['nrhs'] == '1' and r['spectrum'] == 'geometric' and r['seed'] == '9200': print('  ', r['target'], r['n'], r['method_id'], r['device_ready_median_s'], 'host', r['host_e2e_median_s'], 'qual', r['qualified_all_reps'], 'outer', r['median_outer'], 'inner', r['median_inner'])
