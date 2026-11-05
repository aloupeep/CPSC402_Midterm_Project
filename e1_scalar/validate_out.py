#!/usr/bin/env python
"""Post-run validation of the E1 output contract (read-only; prints a report).

Checks: JSON validity without NaN/Infinity, counts, file existence, sha256 of A/b/x files
against the records, reference residual at 100 digits, factor identity flags, qualification
booleans consistent with metrics, history lengths vs outer_iters, audit subset rule.
"""
import sys, json, hashlib, math
from pathlib import Path
import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/worker/lab/campaign/e1_scalar/out')
U32 = 2.0 ** -24
problems = []


def check(cond, msg):
    if not cond:
        problems.append(msg)


raw_cases = (out / 'cases.jsonl').read_text().splitlines()
raw_trials = (out / 'trials.jsonl').read_text().splitlines()
for name, raw in (('cases', raw_cases), ('trials', raw_trials)):
    for i, l in enumerate(raw):
        check('NaN' not in l and 'Infinity' not in l, f'{name} line {i}: NaN/Infinity literal')
        json.loads(l)  # strict parse (will raise on failure)


def strict(s):
    def bad(x):
        raise ValueError('nonfinite literal ' + x)
    return json.loads(s, parse_constant=bad)


cases = [strict(l) for l in raw_cases]
trials = [strict(l) for l in raw_trials]
methods = strict((out / 'methods.json').read_text())
print(f'cases={len(cases)} trials={len(trials)} methods={len(methods)}')
check(len(cases) == 54, f'expected 54 cases, got {len(cases)}')
check(len(trials) == 54 * 8, f'expected 432 trials, got {len(trials)}')
ids = [c['case_id'] for c in cases]
check(len(set(ids)) == len(ids), 'duplicate case ids')
idx_sorted = sorted(cases, key=lambda c: (c['n'], c['kappa_nominal'], c['seed']))
for i, c in enumerate(idx_sorted):
    check(c['case_index_sorted'] == i, f'case index mismatch {c["case_id"]}')
    check((c['audit_digits'] == 120) == (i % 5 == 0), f'audit rule violated for index {i}')
    if c['audit_digits']:
        check(c['reference_audit_maxrel'] is not None and c['reference_audit_maxrel'] < 1e-80, f'audit maxrel large: {c["reference_audit_maxrel"]}')
    check(c['reference_residual_inf_mp100'] < 1e-80, f'reference residual large {c["reference_residual_inf_mp100"]}')
    a = np.load(out / c['a_path'])
    b = np.load(out / c['b_path'])
    check(a.dtype == np.float32 and b.dtype == np.float32 and a.shape == (c['n'], c['n']) and b.shape == (c['n'],), 'A/b dtype/shape')
    check(hashlib.sha256(a.tobytes()).hexdigest() == c['a_sha256'], f'a_sha256 mismatch {c["case_id"]}')
    check(hashlib.sha256(b.tobytes()).hexdigest() == c['b_sha256'], f'b_sha256 mismatch {c["case_id"]}')
    cid2 = hashlib.sha256(a.tobytes() + b.tobytes() + f'|n={c["n"]}|seed={c["seed"]}'.encode()).hexdigest()[:16]
    check(cid2 == c['case_id'], f'case_id derivation mismatch {c["case_id"]}')
    xr = np.load(out / c['reference_path'])
    check(xr.dtype == np.float64 and xr.shape == (c['n'],), 'reference dtype/shape')
    txt = (out / c['reference_txt_path']).read_text().split()
    check(len(txt) == c['n'], 'reference txt length')
    check(all(v for v in c['factor_identity_checks'].values()), f'factor identity check failed {c["case_id"]} {c["factor_identity_checks"]}')
    # the FP64 residual of the reference should be at FP64 rounding level
    r = b.astype(np.float64) - a.astype(np.float64) @ xr
    den = np.linalg.norm(a.astype(np.float64), np.inf) * np.max(np.abs(xr)) + np.max(np.abs(b))
    check(np.max(np.abs(r)) / den < 1e-13, f'reference FP64 backward error suspicious {np.max(np.abs(r)) / den:.2e} {c["case_id"]}')
    check(0.5 < c['kappa2_measured'] / c['kappa_nominal'] < 2.0, f'kappa2 measured {c["kappa2_measured"]:.3g} vs nominal {c["kappa_nominal"]:.3g}')

cid2case = {c['case_id']: c for c in cases}
per_case_methods = {}
for t in trials:
    c = cid2case[t['case_id']]
    per_case_methods.setdefault(t['case_id'], []).append(t['method_id'])
    m = t['metrics']
    q = t['qualification']
    for key, thr in (('eta', 10), ('omega', 10), ('finf', 10), ('eta', 100), ('omega', 100), ('finf', 100)):
        v = m[key]
        expect = bool(v is not None and v <= thr * U32)
        check(q[f'{key}_le_{thr}u'] == expect, f'qualification inconsistency {t["trial_id"]} {key} {thr}')
    for k, v in m.items():
        check(v is None or (isinstance(v, (int, float)) and math.isfinite(v)), f'metric nonfinite in json {t["trial_id"]} {k}')
        if v is None:
            check(t['nonfinite_reason'] is not None and k in t['nonfinite_reason'], f'null metric without reason {t["trial_id"]} {k}')
    if t['x_path']:
        x = np.load(out / t['x_path'])
        check(x.dtype == np.float32, 'x dtype')
        check(hashlib.sha256(x.tobytes()).hexdigest() == t['x_sha256'], f'x_sha256 mismatch {t["trial_id"]}')
        if m['finf'] is not None:
            xr = np.load(out / c['reference_path'])
            finf = np.max(np.abs(x.astype(np.float64) - xr)) / np.max(np.abs(xr))
            check(abs(finf - m['finf']) <= 1e-9 * max(1.0, finf), f'finf recompute mismatch {t["trial_id"]}')
        if m['residual_inf_mp100'] is not None:
            a = np.load(out / c['a_path']); b = np.load(out / c['b_path'])
            r64 = float(np.max(np.abs(b.astype(np.float64) - a.astype(np.float64) @ x.astype(np.float64))))
            if r64 > 1e-30:
                check(abs(r64 - m['residual_inf_mp100']) / r64 < 1e-6, f'mp100 vs fp64 residual mismatch {t["trial_id"]}: {r64} vs {m["residual_inf_mp100"]}')
    h = strict((out / t['history_path']).read_text())
    check(h['trial_id'] == t['trial_id'], 'history id')
    check(len(h['history']) == t['n_iterates_saved'], 'history length vs iterates')
    if t['terminal_reason'] != 'numerical_failure':
        check(len(h['history']) == t['outer_iters'] + 1, f'history length {len(h["history"])} vs outer {t["outer_iters"]} {t["trial_id"]}')
    fa = next((e['correction_index'] for e in h['history'] if e['finf_le_10u']), None)
    check(fa == t['first_attainment_10u'], f'first attainment mismatch {t["trial_id"]}')
    if t['method_id'].startswith('paper_left_gmres'):
        check(t['gmres_params'] is not None and t['gmres_params']['m_used'] == 30, 'gmres params')
        check(len(t['gmres_params']['per_outer']) >= t['outer_iters'], 'gmres per-outer records')
    else:
        check(t['gmres_params'] is None and t['inner_iters_total'] == 0, 'classical gmres fields')
    check(t['factor_hash_verified_against_shared_factorization'], f'factor hash not verified {t["trial_id"]}')
    check(t['factor_sha256'] == (c['half_factor_sha256'] if t['method_id'] in ('hss', 'hsd', 'half_factors_solve32', 'paper_left_gmres_hsd_app64', 'paper_left_gmres_hsd_app32')
                                 else c['fp32_factor_sha256'] if t['method_id'] in ('sss', 'ssd') else c['fp32_factor_sha256_input16']), f'factor_sha256 wrong {t["trial_id"]}')
for cid, ms in per_case_methods.items():
    check(sorted(ms) == sorted(['hss', 'hsd', 'sss', 'ssd', 'input16_ssd', 'half_factors_solve32', 'paper_left_gmres_hsd_app64', 'paper_left_gmres_hsd_app32']), f'method set for {cid}')
# x_0 identity: hsd, hss, half_factors_solve32 and both GMRES methods share the same initial iterate (history index 0 metrics identical)
for cid in cid2case:
    h0 = {}
    # half_factors_solve32 applies the factors in FP32 also for the initial solve (prior-package design), so it is excluded
    for mid in ('hss', 'hsd', 'paper_left_gmres_hsd_app64', 'paper_left_gmres_hsd_app32'):
        h = strict((out / 'histories' / f'{cid}__{mid}.json').read_text())
        if h['history']:
            h0[mid] = (h['history'][0]['finf'], h['history'][0]['eta'])
    check(len(set(h0.values())) <= 1, f'x_0 differs across half-factor methods for {cid}: {h0}')
print('PROBLEMS:', len(problems))
for p in problems[:50]:
    print(' -', p)
