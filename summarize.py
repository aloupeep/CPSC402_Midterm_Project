#!/usr/bin/env python3
"""Regenerate every summary table, figure and the claim ledger from writeup_results/{cases,trials,histories}.jsonl (no GPU needed)."""
from __future__ import annotations
import csv, json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = Path('/home/worker/lab/campaign/writeup_results'); S = OUT / 'summaries'; FIG = OUT / 'figures'; S.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
cases = {json.loads(l)['case_id']: json.loads(l) for l in open(OUT / 'cases.jsonl')}
trials = [json.loads(l) for l in open(OUT / 'trials.jsonl')]
def med(v): v = [x for x in v if x is not None]; return st.median(v) if v else None
def q(v, p): v = sorted(x for x in v if x is not None); return v[min(len(v) - 1, int(p * len(v)))] if v else None
def wcsv(name, rows, cols):
    with open(S / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r.get(c) for c in cols}) for r in rows]
BASE = {'fp64': 'native_lu64', 'fp32': 'native_lu32'}
claims = []
# ---------------------------------------------------------------- 1. qualification by suite/target/family/method
Q = defaultdict(lambda: dict(n=0, pass_primary=0, pass_strict=0, fwd10=0, fwd100=0, fwd_eval=0, nonfinite=0, unsupported=0, memory_skip=0, runtime_error=0, fallback=0, outer=[], inner=[]))
for t in trials:
    if t['run_kind'] == 'warmup' or t['phase'] == 'development': continue
    c = cases[t['case_id']]; fam = c['family'] + ('_' + c['spectrum'] if c.get('spectrum') else ''); key = (t['suite'], t['target_precision'], fam, t['method_id'])
    d = Q[key]; d['n'] += 1; v = t['validation']
    d['pass_primary'] += v['both_backward'] == 'pass'; d['pass_strict'] += v.get('backward_strict_10u') == 'pass'
    if v.get('forward_10u') in ('pass', 'fail', 'indeterminate'): d['fwd_eval'] += 1; d['fwd10'] += v['forward_10u'] == 'pass'; d['fwd100'] += v['forward_100u'] == 'pass'
    d['nonfinite'] += 'nonfinite' in (t['terminal_reason'] or ''); d['unsupported'] += 'unsupported' in (t['terminal_reason'] or ''); d['memory_skip'] += t.get('skipped', False); d['runtime_error'] += (not t.get('ok', True)) and 'unsupported' not in (t['terminal_reason'] or ''); d['fallback'] += bool(t.get('used_fallback'))
    d['outer'].append(t['counters'].get('outer')); d['inner'].append(t['counters'].get('inner_total'))
rows = [dict(suite=k[0], target=k[1], family=k[2], method_id=k[3], trials=d['n'], pass_primary_100u=d['pass_primary'], pass_strict_10u=d['pass_strict'], forward_evaluated=d['fwd_eval'], forward_pass_10u=d['fwd10'], forward_pass_100u=d['fwd100'], nonfinite=d['nonfinite'], unsupported=d['unsupported'], memory_skip=d['memory_skip'], runtime_error=d['runtime_error'], used_fallback=d['fallback'], median_outer=med(d['outer']), median_inner=med(d['inner'])) for k, d in sorted(Q.items())]
wcsv('qualification_by_suite_target_family_method.csv', rows, list(rows[0].keys()) if rows else ['suite'])
# ---------------------------------------------------------------- 2. E2 conditioning map: pass per (target, n, spectrum, kappa, method)
E2 = defaultdict(lambda: dict(n=0, p=0, ps=0, f10=0, fe=0, outer=[], t=[], nf=0))
for t in trials:
    if t['suite'] != 'E2' or t['run_kind'] == 'warmup': continue
    c = cases[t['case_id']]
    if c['family'] != 'prescribed': continue
    d = E2[(t['target_precision'], c['n'], c['spectrum'], c['kappa_nominal'], t['method_id'])]; d['n'] += 1; d['p'] += t['validation']['both_backward'] == 'pass'; d['ps'] += t['validation'].get('backward_strict_10u') == 'pass'
    if t['validation'].get('forward_10u') in ('pass', 'fail', 'indeterminate'): d['fe'] += 1; d['f10'] += t['validation']['forward_10u'] == 'pass'
    d['outer'].append(t['counters']['outer']); d['t'].append(t['timings_s']['device_ready_wall']); d['nf'] += 'nonfinite' in (t['terminal_reason'] or '')
rows = [dict(target=k[0], n=k[1], spectrum=k[2], kappa_nominal=k[3], method_id=k[4], seeds=d['n'], pass_primary=d['p'], pass_strict=d['ps'], forward_evaluated=d['fe'], forward_pass_10u=d['f10'], nonfinite=d['nf'], median_outer=med(d['outer']), median_device_ready_s=med(d['t'])) for k, d in sorted(E2.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4]))]
wcsv('e2_conditioning_map.csv', rows, list(rows[0].keys()) if rows else ['target'])
# ---------------------------------------------------------------- 3. E4 timing per case/method (median over reps) and paired speedups vs frozen baseline
T = defaultdict(lambda: dict(dev=[], host=[], passes=0, n=0, outer=[], inner=[], fallback=0, term=set(), mem=[]))
for t in trials:
    if t['suite'] not in ('E4',) or t['run_kind'] == 'warmup': continue
    if t.get('skipped') or not t.get('ok', True): continue  # memory skips / runtime errors stay in coverage.csv; a later separate-process re-run supplies the timed records
    c = cases[t['case_id']]; key = (t['target_precision'], c['n'], c['nrhs'], c['spectrum'], c['kappa_nominal'], c['seed'], t['method_id']); d = T[key]
    if t['run_kind'] == 'timed_resident' and t['timings_s']['device_ready_wall'] is not None: d['dev'].append(t['timings_s']['device_ready_wall'])
    if t['run_kind'] == 'timed_host' and t['timings_s']['host_e2e'] is not None: d['host'].append(t['timings_s']['host_e2e'])
    d['n'] += 1; d['passes'] += t['validation']['both_backward'] == 'pass'; d['outer'].append(t['counters']['outer']); d['inner'].append(t['counters']['inner_total']); d['fallback'] += bool(t.get('used_fallback')); d['term'].add(t['terminal_reason']); d['mem'].append(t['memory']['peak_device_bytes_tracked'])
rows = []
for k, d in sorted(T.items()):
    rows.append(dict(target=k[0], n=k[1], nrhs=k[2], spectrum=k[3], kappa_nominal=k[4], seed=k[5], method_id=k[6], reps_resident=len(d['dev']), reps_host=len(d['host']), device_ready_median_s=med(d['dev']), device_ready_min_s=min(d['dev']) if d['dev'] else None, device_ready_max_s=max(d['dev']) if d['dev'] else None, device_ready_iqr_s=(q(d['dev'], .75) - q(d['dev'], .25)) if len(d['dev']) >= 2 else None, host_e2e_median_s=med(d['host']), host_e2e_min_s=min(d['host']) if d['host'] else None, qualified_all_reps=(d['passes'] == d['n'] and d['n'] > 0), qualified_count=d['passes'], records=d['n'], median_outer=med(d['outer']), median_inner=med(d['inner']), fallback_count=d['fallback'], terminal_reasons=';'.join(sorted(x for x in d['term'] if x)), peak_device_bytes=max(d['mem']) if d['mem'] else None, raw_device_ready_s=json.dumps(d['dev']), raw_host_e2e_s=json.dumps(d['host'])))
wcsv('e4_timing_per_case_method.csv', rows, list(rows[0].keys()) if rows else ['target'])
# speedups: two denominators. (a) frozen ordinary baseline native_lu64/native_lu32 on ALL cases where the METHOD qualified (baseline qualification shown);
# (b) qualified full-precision-refinement baseline native_ir64_w64_r64 (target64) / native_ir32_w32_r64 (target32) on cases where BOTH qualified.
BASE2 = {'fp64': 'native_ir64_w64_r64', 'fp32': 'native_ir32_w32_r64'}
SP = defaultdict(list); COVR = defaultdict(lambda: [0, 0]); SP2 = defaultdict(list); COVR2 = defaultdict(lambda: [0, 0])
bycase = defaultdict(dict)
for r in rows: bycase[(r['target'], r['n'], r['nrhs'], r['spectrum'], r['kappa_nominal'], r['seed'])][r['method_id']] = r
sprows = []
for ck, ms in sorted(bycase.items()):
    b = ms.get(BASE[ck[0]]); b2 = ms.get(BASE2[ck[0]])
    for m, r in ms.items():
        if m == BASE[ck[0]]: continue
        rec = dict(target=ck[0], n=ck[1], nrhs=ck[2], spectrum=ck[3], kappa_nominal=ck[4], seed=ck[5], method_id=m, method_qualified=r['qualified_all_reps'], method_device_ready_median_s=r['device_ready_median_s'],
                   baseline=BASE[ck[0]], baseline_qualified=(b['qualified_all_reps'] if b else None), baseline_device_ready_median_s=(b['device_ready_median_s'] if b else None), ratio_vs_ordinary_lu=None, ratio_vs_ordinary_lu_host=None,
                   refined_baseline=BASE2[ck[0]], refined_baseline_qualified=(b2['qualified_all_reps'] if b2 else None), refined_baseline_device_ready_median_s=(b2['device_ready_median_s'] if b2 else None), ratio_vs_refined_baseline=None)
        COVR[(ck[0], ck[1], ck[2], m)][1] += 1
        if b and r['qualified_all_reps'] and r['device_ready_median_s'] and b['device_ready_median_s']:
            COVR[(ck[0], ck[1], ck[2], m)][0] += 1; rec['ratio_vs_ordinary_lu'] = b['device_ready_median_s'] / r['device_ready_median_s']; SP[(ck[0], ck[1], ck[2], m)].append(rec['ratio_vs_ordinary_lu'])
            if b['host_e2e_median_s'] and r['host_e2e_median_s']: rec['ratio_vs_ordinary_lu_host'] = b['host_e2e_median_s'] / r['host_e2e_median_s']
        if m != BASE2[ck[0]]:
            COVR2[(ck[0], ck[1], ck[2], m)][1] += 1
            if b2 and r['qualified_all_reps'] and b2['qualified_all_reps'] and r['device_ready_median_s'] and b2['device_ready_median_s']:
                COVR2[(ck[0], ck[1], ck[2], m)][0] += 1; rec['ratio_vs_refined_baseline'] = b2['device_ready_median_s'] / r['device_ready_median_s']; SP2[(ck[0], ck[1], ck[2], m)].append(rec['ratio_vs_refined_baseline'])
        sprows.append(rec)
wcsv('e4_paired_speedups.csv', sprows, list(sprows[0].keys()) if sprows else ['target'])
def gm(v): return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None
agg = []
for k in sorted(set(COVR) | set(COVR2)):
    v = SP.get(k, []); v2 = SP2.get(k, [])
    agg.append(dict(target=k[0], n=k[1], nrhs=k[2], method_id=k[3], method_qualified_cases=COVR[k][0] if k in COVR else None, total_cases=COVR[k][1] if k in COVR else COVR2[k][1], median_ratio_vs_ordinary_lu=med(v), min_ratio_vs_ordinary_lu=min(v) if v else None, max_ratio_vs_ordinary_lu=max(v) if v else None, geomean_ratio_vs_ordinary_lu=gm(v),
                    both_qualified_cases_vs_refined=COVR2[k][0] if k in COVR2 else None, median_ratio_vs_refined_baseline=med(v2), min_ratio_vs_refined_baseline=min(v2) if v2 else None, max_ratio_vs_refined_baseline=max(v2) if v2 else None, geomean_ratio_vs_refined_baseline=gm(v2)))
wcsv('e4_speedup_summary.csv', agg, list(agg[0].keys()) if agg else ['target'])
# ---------------------------------------------------------------- 4. E6 policies
P = defaultdict(lambda: dict(n=0, p=0, t=[], stages=defaultdict(int), inner=[], refac=[], fb=0))
for t in trials:
    if t['suite'] != 'E6' or t['run_kind'] == 'warmup': continue
    c = cases[t['case_id']]; key = (t['phase'], c['n'], c['family'] + ('_k%g' % c['kappa_nominal'] if c['family'] == 'prescribed' else '_rbf'), t['method_id']); d = P[key]
    d['n'] += 1; d['p'] += t['validation']['both_backward'] == 'pass'; d['t'].append(t['timings_s']['device_ready_wall']); d['stages'][t['actual_final_stage']] += 1; d['inner'].append(t['counters']['inner_total']); d['refac'].append(t['counters']['refactorizations']); d['fb'] += bool(t['used_fallback'])
rows = [dict(phase=k[0], n=k[1], case_family=k[2], method_id=k[3], cases=d['n'], pass_primary=d['p'], median_device_ready_s=med(d['t']), max_device_ready_s=max(x for x in d['t'] if x is not None) if any(x is not None for x in d['t']) else None, final_stages=json.dumps(dict(d['stages'])), median_inner=med(d['inner']), median_refactorizations=med(d['refac']), fallback_count=d['fb']) for k, d in sorted(P.items())]
wcsv('e6_policies.csv', rows, list(rows[0].keys()) if rows else ['phase'])
# ---------------------------------------------------------------- 5. E5 reuse
R = defaultdict(lambda: dict(reuse=[], fresh=[], factor=[], p=0, n=0))
for t in trials:
    if t['suite'] != 'E5' or t['run_kind'] == 'warmup': continue
    c = cases[t['case_id']]; d = R[(c['n'], c['nrhs'], t['method_id'])]
    (d['reuse'] if t['run_kind'] == 'reuse' else d['fresh']).append(t['timings_s']['device_ready_wall']); d['n'] += 1; d['p'] += t['validation']['both_backward'] == 'pass'
    if t['run_kind'] != 'reuse' and t['timings_s'].get('factor_total'): d['factor'].append(t['timings_s']['factor_total'])
rows = [dict(n=k[0], nrhs=k[1], method_id=k[2], reuse_solve_median_s=med(d['reuse']), fresh_solve_median_s=med(d['fresh']), factor_median_s=med(d['factor']), qualified=d['p'], records=d['n'], per_rhs_reuse_s=(med(d['reuse']) / k[1]) if med(d['reuse']) else None) for k, d in sorted(R.items())]
wcsv('e5_reuse.csv', rows, list(rows[0].keys()) if rows else ['n'])
# ---------------------------------------------------------------- 6. E3 scaling
X = defaultdict(lambda: dict(n=0, p=0, nf=0, outer=[], fb=0))
for t in trials:
    if t['suite'] != 'E3' or t['run_kind'] == 'warmup': continue
    c = cases[t['case_id']]; d = X[(c['n'], c['scaling']['global_exp'], c['scaling']['rowcol_span'], t['method_id'])]; d['n'] += 1; d['p'] += t['validation']['both_backward'] == 'pass'; d['nf'] += 'nonfinite' in (t['terminal_reason'] or ''); d['outer'].append(t['counters']['outer']); d['fb'] += bool(t['used_fallback'])
rows = [dict(n=k[0], global_exp=k[1], rowcol_span=k[2], method_id=k[3], seeds=d['n'], pass_primary=d['p'], nonfinite=d['nf'], median_outer=med(d['outer']), fallback=d['fb']) for k, d in sorted(X.items())]
wcsv('e3_scaling.csv', rows, list(rows[0].keys()) if rows else ['n'])
# ---------------------------------------------------------------- 7. memory, GMRES work, profiles
rows = [dict(target=t['target_precision'], n=cases[t['case_id']]['n'], nrhs=cases[t['case_id']]['nrhs'], method_id=t['method_id'], peak_device_bytes_tracked=t['memory']['peak_device_bytes_tracked'], device_free_after_bytes=t['memory']['device_free_after_bytes'], irs_workspace_bytes=t['memory'].get('irs_workspace_bytes'), stored_matrix_bytes=cases[t['case_id']]['memory_plan']['stored_matrix_bytes']) for t in trials if t['suite'] == 'E4' and t['run_kind'] == 'timed_resident' and t['rep'] == 0]
wcsv('memory_e4.csv', rows, list(rows[0].keys()) if rows else ['target'])
rows = [dict(suite=t['suite'], n=cases[t['case_id']]['n'], case_id=t['case_id'], method_id=t['method_id'], run_kind=t['run_kind'], outer=t['counters']['outer'], inner_total=t['counters']['inner_total'], restarts=t['counters']['restarts'], matvecs=t['counters']['matvecs'], precond=t['counters']['preconditioner_applications'], true_checks=t['counters']['true_residual_checks'], terminal=t['terminal_reason'], device_ready_s=t['timings_s']['device_ready_wall'], qualified=t['validation']['both_backward']) for t in trials if t['run_kind'] != 'warmup' and t['counters'].get('inner_total') and t['counters']['inner_total'] > 0 and t['suite'] != 'E1']
wcsv('gmres_work.csv', rows, list(rows[0].keys()) if rows else ['suite'])
rows = [dict(case_id=t['case_id'], n=cases[t['case_id']]['n'], method_id=t['method_id'], device_ready_s=t['timings_s']['device_ready_wall'], factor_s=t['timings_s']['factor_total'], initial_solve_s=t['timings_s']['initial_solve'], refine_s=t['timings_s']['refine'], convergence_checks_s=t['timings_s']['internal_convergence_checks'], outer=t['counters']['outer'], inner=t['counters']['inner_total'], stages=json.dumps(t['counters']['stages']), stage_times=json.dumps(t['counters']['stage_times_s'])) for t in trials if t['run_kind'] == 'profile']
wcsv('profiles_stage_breakdown.csv', rows, list(rows[0].keys()) if rows else ['case_id'])
# ---------------------------------------------------------------- 8. E1 tables (from the scalar track)
e1 = defaultdict(lambda: dict(n=0, p=0, f10=0, f100=0, nf=0))
for t in trials:
    if t['suite'] != 'E1': continue
    c = cases[t['case_id']]; d = e1[(t['method_id'], c['kappa_nominal'])]; d['n'] += 1; d['p'] += t['validation']['both_backward'] == 'pass'; d['f10'] += t['validation']['forward_10u'] == 'pass'; d['f100'] += t['validation']['forward_100u'] == 'pass'; d['nf'] += bool(t['metrics'].get('nonfinite_reason'))
rows = [dict(method_id=k[0], kappa_nominal=k[1], cases=d['n'], backward_pass_100u=d['p'], forward_pass_10u=d['f10'], forward_pass_100u=d['f100'], nonfinite=d['nf']) for k, d in sorted(e1.items())]
wcsv('e1_scalar_qualification.csv', rows, list(rows[0].keys()) if rows else ['method_id'])
# ---------------------------------------------------------------- figures
def fig_time_vs_n():
    for target in ('fp64', 'fp32'):
        sub = [r for r in sprows if r['target'] == target] ; byn = defaultdict(lambda: defaultdict(list))
        for r in [x for x in csv.DictReader(open(S / 'e4_timing_per_case_method.csv'))]:
            if r['target'] != target or r['nrhs'] != '1' or r['spectrum'] != 'geometric' or not r['device_ready_median_s']: continue
            byn[r['method_id']][int(r['n'])].append((float(r['device_ready_median_s']), r['qualified_all_reps'] == 'True'))
        if not byn: continue
        plt.figure(figsize=(8, 5.5))
        for m, dd in sorted(byn.items()):
            ns = sorted(dd); ys = [st.median([v for v, _ in dd[n]]) for n in ns]; ok = [all(qq for _, qq in dd[n]) for n in ns]
            plt.plot(ns, ys, '-', alpha=.6, label=m); plt.scatter([n for n, o in zip(ns, ok) if o], [y for y, o in zip(ys, ok) if o], marker='o', s=22); plt.scatter([n for n, o in zip(ns, ok) if not o], [y for y, o in zip(ys, ok) if not o], marker='x', s=40, c='k')
        plt.xscale('log', base=2); plt.yscale('log'); plt.xlabel('n'); plt.ylabel('device-ready wall time [s] (median over reps, median over seeds)'); plt.title(f'E4 target{target[2:]}: geometric kappa=1e2, nrhs=1 (x = not qualified at 100u, primary profile)'); plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG / f'e4_time_vs_n_{target}.png', dpi=130); plt.close()
def fig_speedup():
    for target in ('fp64', 'fp32'):
        rows = [r for r in agg if r['target'] == target and r['nrhs'] == 1 and (r['median_ratio_vs_ordinary_lu'] or r['median_ratio_vs_refined_baseline'])]
        if not rows: continue
        ms = sorted({r['method_id'] for r in rows}); plt.figure(figsize=(8, 5.5))
        for m in ms:
            rr = sorted([r for r in rows if r['method_id'] == m and r['median_ratio_vs_ordinary_lu']], key=lambda r: r['n'])
            if rr: plt.plot([r['n'] for r in rr], [r['median_ratio_vs_ordinary_lu'] for r in rr], 'o-', label=f"{m} ({sum(r['method_qualified_cases'] or 0 for r in rr)}/{sum(r['total_cases'] for r in rr)} method-qualified)")
        plt.axhline(1, c='k', lw=.8); plt.xscale('log', base=2); plt.xlabel('n'); plt.ylabel(f'T({BASE[target]}) / T(method), device-ready; method qualified at 100u, baseline qualification in table'); plt.title(f'E4 target{target[2:]}: nrhs=1 timing ratios vs ordinary LU (see also ratios vs refined baseline)'); plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG / f'e4_speedup_vs_n_{target}.png', dpi=130); plt.close()
def fig_e2():
    rows = list(csv.DictReader(open(S / 'e2_conditioning_map.csv')))
    for target in ('fp64', 'fp32'):
        sub = [r for r in rows if r['target'] == target and r['n'] == '2048' and r['spectrum'] == 'geometric']
        if not sub: continue
        ms = sorted({r['method_id'] for r in sub}); ks = sorted({float(r['kappa_nominal']) for r in sub}); plt.figure(figsize=(9, 5))
        for i, m in enumerate(ms):
            ys = []
            for k in ks:
                rr = [r for r in sub if r['method_id'] == m and float(r['kappa_nominal']) == k]; ys.append(int(rr[0]['pass_primary']) / int(rr[0]['seeds']) if rr else float('nan'))
            plt.plot(ks, [y + 0.004 * i for y in ys], 'o-', ms=3, label=m)
        plt.xscale('log'); plt.xlabel('nominal kappa2'); plt.ylabel('fraction of seeds qualified (100u primary profile)'); plt.title(f'E2 target{target[2:]}: geometric spectrum, n=2048, 3 seeds'); plt.legend(fontsize=5, ncol=2); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG / f'e2_qualification_vs_kappa_{target}.png', dpi=130); plt.close()
def fig_breakdown():
    rows = list(csv.DictReader(open(S / 'profiles_stage_breakdown.csv')))
    if not rows: return
    plt.figure(figsize=(9, 5)); labels = [f"{r['method_id']}\nn={r['n']}" for r in rows]; f = [float(r['factor_s'] or 0) for r in rows]; ini = [float(r['initial_solve_s'] or 0) for r in rows]; ref = [float(r['refine_s'] or 0) for r in rows]; chk = [float(r['convergence_checks_s'] or 0) for r in rows]
    x = range(len(rows)); plt.bar(x, f, label='factorization (incl. conversion/scaling)'); plt.bar(x, ini, bottom=f, label='initial solve'); plt.bar(x, [r - c for r, c in zip(ref, chk)], bottom=[a + b for a, b in zip(f, ini)], label='refinement (corrections)'); plt.bar(x, chk, bottom=[a + b + r - c for a, b, r, c in zip(f, ini, ref, chk)], label='convergence checks (residual+metrics)')
    plt.xticks(list(x), labels, fontsize=6, rotation=30); plt.ylabel('seconds (instrumented run, E8)'); plt.title('Stage breakdown (instrumented profile runs; sequential stage timers, not kernel-level)'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / 'e8_stage_breakdown.png', dpi=130); plt.close()
def fig_e6():
    rows = list(csv.DictReader(open(S / 'e6_policies.csv')))
    sub = [r for r in rows if r['phase'] == 'evaluation' and r['case_family'] == 'prescribed_k1e+06']
    if not sub: return
    ms = sorted({r['method_id'] for r in sub}); plt.figure(figsize=(8, 5))
    for m in ms:
        rr = sorted([r for r in sub if r['method_id'] == m], key=lambda r: int(r['n'])); plt.plot([int(r['n']) for r in rr], [float(r['median_device_ready_s']) if r['median_device_ready_s'] else float('nan') for r in rr], 'o-', label=f"{m} (pass {sum(int(r['pass_primary']) for r in rr)}/{sum(int(r['cases']) for r in rr)})")
    plt.xscale('log', base=2); plt.yscale('log'); plt.xlabel('n'); plt.ylabel('median device-ready time to termination [s]'); plt.title('E6 recovery policies, geometric kappa=1e6, target64 (3 seeds)'); plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG / 'e6_policy_cost_vs_n.png', dpi=130); plt.close()
def fig_reuse():
    rows = list(csv.DictReader(open(S / 'e5_reuse.csv')))
    if not rows: return
    plt.figure(figsize=(8, 5))
    for m in sorted({r['method_id'] for r in rows}):
        rr = sorted([r for r in rows if r['method_id'] == m and r['n'] == '8192' and r['reuse_solve_median_s']], key=lambda r: int(r['nrhs']))
        if rr: plt.plot([int(r['nrhs']) for r in rr], [float(r['reuse_solve_median_s']) for r in rr], 'o-', label=m)
    plt.xscale('log', base=2); plt.yscale('log'); plt.xlabel('nrhs (fresh RHS block, factors reused)'); plt.ylabel('solve+refinement time with reused factors [s]'); plt.title('E5 factor reuse, n=8192, geometric kappa=1e2, target64'); plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG / 'e5_reuse_n8192.png', dpi=130); plt.close()
def fig_scatter():
    pts = [(t['timings_s']['device_ready_wall'], max(x for x in t['metrics']['normwise_backward_per_rhs'] if x is not None), t['method_id']) for t in trials if t['suite'] == 'E2' and t['run_kind'] != 'warmup' and cases[t['case_id']]['n'] == 2048 and t['target_precision'] == 'fp64' and t['timings_s']['device_ready_wall'] and any(x is not None for x in t['metrics']['normwise_backward_per_rhs'])]
    if not pts: return
    plt.figure(figsize=(8, 5.5)); ms = sorted({p[2] for p in pts})
    for m in ms: plt.scatter([p[0] for p in pts if p[2] == m], [p[1] for p in pts if p[2] == m], s=10, label=m, alpha=.7)
    plt.axhline(100 * 2 ** -53, c='k', ls='--', lw=.8); plt.xscale('log'); plt.yscale('log'); plt.xlabel('device-ready time [s]'); plt.ylabel('normwise backward error eta (max over RHS)'); plt.title('E2 time-accuracy scatter, n=2048, FP64 inputs (dashed: 100 u64)'); plt.legend(fontsize=5, ncol=2); plt.tight_layout(); plt.savefig(FIG / 'e2_time_accuracy_scatter_n2048_fp64.png', dpi=130); plt.close()
def fig_memory():
    rows = list(csv.DictReader(open(S / 'memory_e4.csv')))
    if not rows: return
    plt.figure(figsize=(8, 5))
    for m in sorted({r['method_id'] for r in rows}):
        rr = sorted([r for r in rows if r['method_id'] == m and r['nrhs'] == '1' and r['target'] == 'fp64'], key=lambda r: int(r['n']))
        if rr: plt.plot([int(r['n']) for r in rr], [int(r['peak_device_bytes_tracked']) / 2 ** 30 for r in rr], 'o-', label=m, ms=3)
    plt.axhline(32, c='k', ls='--', lw=.8); plt.xscale('log', base=2); plt.xlabel('n'); plt.ylabel('peak tracked device allocation [GiB] (process-level, includes stored A and reference buffers)'); plt.title('Memory feasibility, FP64 inputs, nrhs=1'); plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG / 'memory_feasibility_fp64.png', dpi=130); plt.close()
def fig_e1():
    rows = list(csv.DictReader(open(S / 'e1_scalar_qualification.csv')))
    if not rows: return
    plt.figure(figsize=(8, 5))
    for m in sorted({r['method_id'] for r in rows}):
        rr = sorted([r for r in rows if r['method_id'] == m], key=lambda r: float(r['kappa_nominal'])); plt.plot([float(r['kappa_nominal']) for r in rr], [int(r['forward_pass_10u']) / int(r['cases']) for r in rr], 'o-', label=m)
    plt.xscale('log'); plt.xlabel('nominal kappa2'); plt.ylabel('fraction with forward error <= 10 u32 (100-digit reference)'); plt.title('E1 scalar three-precision model, n in {64,128,256}, 3 seeds'); plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG / 'e1_forward_qualification_vs_kappa.png', dpi=130); plt.close()
for f in (fig_time_vs_n, fig_speedup, fig_e2, fig_breakdown, fig_e6, fig_reuse, fig_scatter, fig_memory, fig_e1):
    try: f()
    except Exception as e: print('figure failed', f.__name__, e)
# ---------------------------------------------------------------- claims ledger (descriptive; evidence paths)
def add_claim(text, scope, evidence, count=None, grade=None, caveats=None): claims.append(dict(claim=text, scope=scope, evidence=evidence, qualification_count=count, reference_grade=grade, caveats=caveats or []))
for r in agg:
    if r['median_ratio_vs_ordinary_lu'] is not None:
        add_claim(f"On E4 target{r['target'][2:]} cases with n={r['n']}, nrhs={r['nrhs']}, method {r['method_id']} (qualified at the primary 100u backward profile on {r['method_qualified_cases']} of {r['total_cases']} cases) had median device-ready time ratio T({BASE[r['target']]})/T(method) = {r['median_ratio_vs_ordinary_lu']:.3f} (min {r['min_ratio_vs_ordinary_lu']:.3f}, max {r['max_ratio_vs_ordinary_lu']:.3f}); the ordinary LU denominator's own qualification is recorded per case in e4_paired_speedups.csv.",
                  dict(suite='E4', target=r['target'], n=r['n'], nrhs=r['nrhs'], families='prescribed geometric kappa=1e2 and clustered kappa=1e4 (as run)', method=r['method_id'], baseline=BASE[r['target']]), ['summaries/e4_speedup_summary.csv', 'summaries/e4_paired_speedups.csv', 'summaries/e4_timing_per_case_method.csv', 'trials.jsonl'], r['method_qualified_cases'], 'backward only (forward reported separately)', ['few seeds; repeated timings are not independent samples; see LIMITATIONS.md'])
    if r['median_ratio_vs_refined_baseline'] is not None:
        add_claim(f"On E4 target{r['target'][2:]} cases with n={r['n']}, nrhs={r['nrhs']}, method {r['method_id']} had median device-ready time ratio T({BASE2[r['target']]})/T(method) = {r['median_ratio_vs_refined_baseline']:.3f} (min {r['min_ratio_vs_refined_baseline']:.3f}, max {r['max_ratio_vs_refined_baseline']:.3f}) over {r['both_qualified_cases_vs_refined']} case(s) where both qualified at the primary profile.",
                  dict(suite='E4', target=r['target'], n=r['n'], nrhs=r['nrhs'], method=r['method_id'], baseline=BASE2[r['target']]), ['summaries/e4_speedup_summary.csv', 'summaries/e4_paired_speedups.csv'], r['both_qualified_cases_vs_refined'], 'backward only', ['few seeds'])
for r in csv.DictReader(open(S / 'qualification_by_suite_target_family_method.csv')):
    add_claim(f"Suite {r['suite']} target{r['target'][2:]} family {r['family']}: method {r['method_id']} met the primary 100u backward profile on {r['pass_primary_100u']} of {r['trials']} evaluated records (strict 10u: {r['pass_strict_10u']}; forward 10u: {r['forward_pass_10u']} of {r['forward_evaluated']} with a reference).", dict(suite=r['suite'], target=r['target'], family=r['family'], method=r['method_id']), ['summaries/qualification_by_suite_target_family_method.csv', 'trials.jsonl'], int(r['pass_primary_100u']), 'per trial (see validation.reference_grade)', [])
json.dump(claims, open(S / 'claims.json', 'w'), indent=1)
print('summaries written:', len(claims), 'claims')
