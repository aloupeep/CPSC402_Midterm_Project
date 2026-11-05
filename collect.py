#!/usr/bin/env python3
"""Collect run directories into the writeup_results/ contract (cases, trials, histories, coverage, solutions, references, env, source, build, logs)."""
from __future__ import annotations
import csv, glob, hashlib, json, math, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, '/home/worker/lab/campaign')
from methods_catalog import CATALOG

ROOT = Path('/home/worker/lab/campaign'); RUNS = ROOT / 'runs'; OUT = ROOT / 'writeup_results'
U32, U64 = 2.0 ** -24, 2.0 ** -53
SUITE_KIND = {'E2a': 'diagnostic', 'dev': 'timed_resident', 'E2': 'timed_resident', 'E3': 'timed_resident', 'E4': 'timed_resident', 'E5': 'timed_resident', 'E6': 'timed_resident', 'E8': 'profile'}

def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 24), b''): h.update(ch)
    return h.hexdigest()

def source_hash():
    files = sorted(glob.glob(str(ROOT / 'src' / '*.cu')) + glob.glob(str(ROOT / 'src' / '*.cuh')) + [str(ROOT / 'campaign.py'), str(ROOT / 'collect.py'), str(ROOT / 'methods_catalog.py'), str(ROOT / 'build' / 'build.sh')] + sorted(glob.glob(str(ROOT / 'e1_scalar' / '*.py'))))
    h = hashlib.sha256()
    for f in files: h.update(Path(f).name.encode()); h.update(open(f, 'rb').read())
    return h.hexdigest()

def fnum(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) or v == -1 else v

def clean_list(v):
    return [None if (x is None or (isinstance(x, float) and not math.isfinite(x))) else x for x in (v or [])]

def qual(vals, tau):
    vals = clean_list(vals)
    if not vals or any(x is None for x in vals): return 'fail' if vals else 'not_evaluated'
    return 'pass' if all(x <= tau for x in vals) else 'fail'

def trial_id(case_id, method, phase, run_kind, rep, tag, mode):
    return f"{case_id}__{method}__{phase}__{run_kind}__{mode}__r{rep}__{tag}"

def main():
    frozen_sha = open(ROOT / 'configs' / 'frozen_config.sha256').read().strip(); src_sha = source_hash()
    audits = {}  # case_id -> audit json (regen + per-solution) produced by audit.py
    if (ROOT / 'audit' / 'audit_results.json').exists(): audits = json.load(open(ROOT / 'audit' / 'audit_results.json'))
    for d in ['env', 'configs', 'source', 'build', 'tests', 'logs', 'matrices', 'matrices/recipes', 'rhs', 'solutions', 'references', 'profiles', 'summaries', 'figures']: (OUT / d).mkdir(parents=True, exist_ok=True)
    cases, trials, hists, cov = {}, [], [], {}
    sol_index = {}
    for suite_dir in sorted(RUNS.iterdir()):
        suite = suite_dir.name
        if suite not in SUITE_KIND: continue
        for cdir in sorted(suite_dir.iterdir()):
            if not cdir.is_dir(): continue
            jobs = [json.load(open(m)) for m in sorted(cdir.glob('done_*.json'))]
            if not jobs: continue
            cfiles = sorted(cdir.glob('case_*.json'))
            cid = cdir.name
            cj = json.load(open(cfiles[0])) if cfiles else None
            job0 = jobs[0]['job']
            tag2job = {j['job']['tag']: j for j in jobs}
            # ---- case record
            if cj and cid not in cases:
                c = job0['case']; n = cj['n']; bytes_el = 4 if cj['dtype'] == 'fp32' else 8
                bsrc = next(cdir.glob('B_*')); bdst = OUT / 'rhs' / (cid + bsrc.suffix); shutil.copyfile(bsrc, bdst)
                recipe = dict(generator='mpsolve generate_problem (source/src/mp_core.cuh)', generator_source_sha256=src_sha, cli_case_args=c, description={
                    'prescribed': 'A = R1^T diag(sigma) R2 where R1,R2 are products of ceil(log2 n)+4 stages of random pairwise Givens rotations (seeded xoshiro256**, Fisher-Yates pairing, angles uniform in [0,2pi)); generated in FP64 on the GPU with fixed-order FMA arithmetic, then rounded to the stored dtype (storage rounding is part of the recipe). Structured random orientation, NOT Haar.',
                    'rbf': 'K_ij = exp(-|x_i-x_j|^2/(2 ell^2)) + nugget*delta_ij, points uniform in [0,1]^dim from seeded xoshiro; FP64 then rounded to stored dtype',
                    'dyadic': 'A = 2^-s C, C_ij integer in [-m,m] from a 64-bit hash of (seed,i,j); b = 2^-s (C x*) with x* = +/-1 from hash(seed^A5A5,j); integer row sums accumulated in int64 (exact); bound n*m <= 2^24 asserted for FP32',
                    'grcar': 'Grcar matrix: -1 on the first subdiagonal, +1 on the diagonal and k superdiagonals (k=grcar_k)',
                    'wilkinson': 'Wilkinson pivot-growth matrix: 1 on the diagonal, -1 strictly below, last column all ones'}[c['family']],
                    rhs={'random': 'standard normal entries from seeded xoshiro256** (seed*104729+0xBEEF+nrhs), rounded to the stored dtype', 'random_scaled': 'normal entries scaled by 2^(4c-2nrhs) per column c', 'exact_sign': 'b = A x* exactly (dyadic family)', 'unit_column': 'b_c = stored column j_c of A (sparse exact solution e_j)'}[c['rhs']],
                    scaling=dict(global_exp=c['global_exp'], rowcol_span=c['rowcol_span'], note='global: A,B multiplied by 2^global_exp (exact); rowcol: A <- D_r A D_c with power-of-two exponents uniform in [-span/2, span/2]'),
                    regen_command=f"build/mpaudit --case-id {cid} " + ' '.join(f"--{k.replace('_','-')} {v}" for k, v in c.items()))
                json.dump(recipe, open(OUT / 'matrices' / 'recipes' / (cid + '.json'), 'w'), indent=1)
                au = audits.get(cid, {})
                embedded = (OUT / 'matrices' / (cid + ('.f32' if cj['dtype'] == 'fp32' else '.f64'))).exists()
                inputs = dict(mode='embedded' if embedded else 'regenerable', b_path=f'rhs/{bdst.name}', recipe_path=f'matrices/recipes/{cid}.json', regeneration_verified=bool(au.get('regenerated_a_sha256') == cj['a_sha256']), regenerated_a_sha256=au.get('regenerated_a_sha256'), regenerated_tile_sha256=au.get('regenerated_tile_sha256'), regeneration_process='separate process (build/mpaudit), see audit/')
                if embedded: inputs['a_path'] = f"matrices/{cid}{'.f32' if cj['dtype']=='fp32' else '.f64'}"
                refs = {}
                for rf in cdir.glob('Xref_*'): shutil.copyfile(rf, OUT / 'references' / f'{cid}__Xref.f64'); refs['numerical_reference_path'] = f'references/{cid}__Xref.f64'
                for rf in cdir.glob('Xexact_*'): shutil.copyfile(rf, OUT / 'references' / f'{cid}__Xexact.f64'); refs['exact_reference_path'] = f'references/{cid}__Xexact.f64'
                cases[cid] = dict(case_id=cid, suite=suite, n=n, nrhs=cj['nrhs'], family=c['family'], spectrum=c['spectrum'] if c['family'] == 'prescribed' else None, seed=c['seed'], dtype=cj['dtype'], rhs=c['rhs'], kappa_nominal=c['kappa'] if c['family'] == 'prescribed' else None,
                                  family_params={k: c[k] for k in ('ell', 'nugget', 'dim', 'intmax', 'dyadic_exp', 'grcar_k')}, scaling=dict(global_exp=c['global_exp'], rowcol_span=c['rowcol_span']),
                                  conditioning_measured=dict(kappa2_estimate=fnum(cj.get('kappa2_estimate')), sigma_max_estimate=fnum(cj.get('sigma_max_estimate')), sigma_min_estimate=fnum(cj.get('sigma_min_estimate')), method=cj.get('kappa_method'), norm='2-norm estimate (power/inverse iteration, 12 steps); NOT a certified condition number'),
                                  anorm_inf=cj['anorm_inf'], amax=cj['amax'], stored_finite=cj['stored_finite'], bnorm_inf=cj['bnorm_inf'], bnorm_2=cj['bnorm_2'],
                                  a_sha256=cj['a_sha256'], b_sha256=cj['b_sha256'], tile_sha256=cj['tile_sha256'], hash_order='SHA-256 of the raw little-endian column-major bytes of the stored-dtype array; tiles = 16 contiguous column blocks [n*t/16, n*(t+1)/16)',
                                  generator=dict(name='mpsolve generate_problem', version=src_sha, config=c), inputs=inputs,
                                  reference=dict(grade='exact_stored' if cj.get('has_exact_solution') else cj['reference_grade'], numerical_grade=cj['reference_grade'], method=cj.get('reference_method'), resid_dd_rel=fnum(cj.get('reference_resid_dd_rel')), resid_dd_abs=fnum(cj.get('reference_resid_dd_abs')), iters=cj.get('reference_iters'), seconds=cj.get('reference_seconds'), note=cj.get('reference_note'), reference_vs_exact_finf=clean_list(cj.get('reference_vs_exact_finf')), **refs),
                                  memory_plan=dict(stored_matrix_bytes=n * n * bytes_el, fp32_factor_bytes=n * n * 4, fp64_factor_bytes=n * n * 8, rhs_bytes=n * cj['nrhs'] * 8, admission_budget_bytes=cj.get('mem_budget_bytes')), generation_seconds=cj['gen_seconds'], hash_seconds=cj['hash_seconds'], target=cj['target'], tau=cj['tau'])
            # ---- trials
            for tf in sorted(cdir.glob('trials_*.jsonl')):
                for line in open(tf):
                    d = json.loads(line)
                    jb = tag2job.get(d['tag'], jobs[0]); j = jb['job']
                    base_kind = j.get('run_kind') or SUITE_KIND[suite]
                    run_kind = 'warmup' if d['warmup'] else ('timed_host' if d['mode'] == 'host' and base_kind in ('timed_resident', 'timed_host') else base_kind)
                    phase = j['phase']; tid = trial_id(cid, d['method_id'], phase, run_kind, d['rep'], d['tag'], d['mode'])
                    tau = d['tau']; u = U32 if d['target'] == 'fp32' else U64
                    finite = d.get('metrics_finite', False) and d['ok'] and not d['skipped']
                    eta, om = clean_list(d['eta']), clean_list(d['omega'])
                    bb = qual(eta, tau) if finite else ('not_evaluated' if (d['skipped'] or not d['ok']) else 'fail')
                    if bb == 'pass' and qual(om, tau) != 'pass': bb = 'fail'
                    if d['warmup']: bb = 'not_evaluated'  # warmup records keep metrics but are excluded from qualification (no saved X)
                    case = cases.get(cid, {}); ref = case.get('reference', {})
                    grade = ref.get('grade', 'unavailable')
                    use_exact = bool(d.get('has_exact')) and grade == 'exact_stored'
                    finf = clean_list(d['einf'] if use_exact else d['finf']) if (d.get('has_exact') or d.get('has_fwd')) else []
                    f2 = clean_list(d['e2'] if use_exact else d['f2']) if finf else []
                    fcomp = clean_list(d['ecomp_nonzero'] if use_exact else d['fcomp_nonzero']) if finf else []
                    ref_unc = None
                    if grade == 'numerical_reference' and ref.get('resid_dd_rel') is not None and case.get('conditioning_measured', {}).get('kappa2_estimate'):
                        ref_unc = ref['resid_dd_rel'] * case['conditioning_measured']['kappa2_estimate']
                    def fq(mult):
                        if not finf or not finite or d['warmup']: return 'not_evaluated'
                        q = qual(finf, mult * u)
                        if grade == 'numerical_reference' and ref_unc is not None and ref_unc > 0.1 * mult * u and q == 'pass': return 'indeterminate'
                        return q
                    term = d['terminal_reason']
                    if d['ok'] and not d['skipped'] and not d.get('metrics_finite', False): term = f"nonfinite_output({term})"
                    if 'status 21' in (d.get('error') or ''): term = 'unsupported_nrhs_for_vendor_gmres'
                    timings = dict(host_e2e=fnum(d['host_e2e_s']) if d['mode'] == 'host' else None, device_ready_wall=fnum(d['device_ready_wall_s']), method_internal_total=fnum(d['method_internal_total_s']), factor_total=fnum(d['factor_total_s']) if d['factor_total_s'] > 0 else None, initial_solve=fnum(d['initial_solve_s']) if d['initial_solve_s'] > 0 else None, refine=fnum(d['refine_s']) if d['refine_s'] > 0 else None, internal_convergence_checks=fnum(d['eval_s']) if d['eval_s'] > 0 else None, wall_including_external_eval=fnum(d['wall_including_eval_s']))
                    sol = None; xs = d.get('x_sha256') or None
                    if d.get('x_path') and xs:
                        ext = '.f32' if d['x_dtype'] == 'float32' else '.f64'; dst = OUT / 'solutions' / (xs + ext)
                        if not dst.exists(): shutil.copyfile(cdir / d['x_path'], dst)
                        sol = f'solutions/{xs}{ext}'
                    au_sol = audits.get(cid, {}).get('solutions', {}).get(xs) if xs else None
                    _per_rhs_keys = ['normwise_backward_per_rhs', 'componentwise_backward_per_rhs', 'forward_inf_per_rhs', 'forward_2_per_rhs', 'relative_residual_inf_per_rhs', 'relative_residual_2_per_rhs']
                    trials.append(dict(trial_id=tid, case_id=cid, method_id=d['method_id'], suite=suite, phase=phase, rep=d['rep'] if d['rep'] >= 0 else 0, run_kind=run_kind, mode=d['mode'], tag=d['tag'], source_sha256=src_sha, source_binary_sha256_note='see build/BUILD_HISTORY.md', config_sha256=frozen_sha if phase in ('evaluation', 'confirmation') else hashlib.sha256(json.dumps(j['flags'], sort_keys=True).encode()).hexdigest(),
                                       dev_flags=j['flags'] if phase == 'development' else None, timestamp=d['timestamp'], method_order_seed=1, environment_id='env/capabilities.json',
                                       target_profile=dict(id='target32_primary_100u' if d['target'] == 'fp32' else 'target64_primary_100u', dtype=d['target'], tau_normwise=tau, tau_componentwise=tau, u=u, strict_tau=10 * u), target_precision=d['target'], working_precision=f"fp{d['working']}",
                                       terminal_reason=term, external_validation=dict(evaluator='driver FP64 k_resid on the original stored system after the solve (excluded from timing)', independent_audit=au_sol), actual_final_stage=d['final_stage'] or 'unknown', used_fallback=bool(d['used_fallback']), skipped=d['skipped'], skip_reason=d['skip_reason'] or None, ok=d['ok'], error=d['error'] or None,
                                       timings_s=timings, timing_boundary=dict(host_e2e='pinned host A,B -> H2D copy -> method (restoration copy, conversion, scaling, factorization, solves, refinement, internal checks) -> D2H X; synchronized steady_clock', device_ready_wall='A,B resident; workspaces provisioned; includes restoration copy of A into the factor buffer, conversions, factorization, solve, refinement, internal convergence checks; synchronized'), warmup=d['warmup'], profiled=bool(j['flags'].get('instrument')), timed_region_policy='mandatory internal convergence checks included; external validation excluded; no CUDA_LAUNCH_BLOCKING',
                                       counters=dict(outer=d['outer_iters'], inner_total=d['inner_iters_total'] if d['inner_iters_total'] >= 0 else None, matvecs=d['matvecs'], preconditioner_applications=d['precond_applies'], triangular_solves=d['triangular_solves'], restarts=d['restarts'], true_residual_checks=d['true_residual_checks'], refactorizations=d['refactorizations'], stages=d['stages'], stage_times_s=d['stage_times_s'], irs_niters=d['irs_niters'] if d['irs_niters'] != -9999 else None, irs_outer_niters=d['irs_outer_niters'] if d['irs_outer_niters'] != -9999 else None, magma_iter=d['magma_iter'] if d['magma_iter'] != -9999 else None, magma_facto_time_s=fnum(d['magma_facto_time_s'])),
                                       factor_diagnostics=dict(info=d['factor_info'], growth_proxy=fnum(d['growth_proxy']), conversion_overflow=d['conv_overflow'], conversion_subnormal=d['conv_subnormal'], scale_mode=d['scale_mode'], nb=d['nb'], panel=d.get('panel')),
                                       metrics=dict(normwise_backward_per_rhs=eta, componentwise_backward_per_rhs=om, forward_inf_per_rhs=finf, forward_2_per_rhs=f2, forward_componentwise_nonzero_per_rhs=fcomp, relative_residual_inf_per_rhs=clean_list(d['relres_inf']), relative_residual_2_per_rhs=clean_list(d['relres_2']), residual_inf_per_rhs=clean_list(d['rinf']), residual_2_per_rhs=clean_list(d['r2']), x_inf_per_rhs=clean_list(d['xinf']), worst_row_per_rhs=d['worst_row'], eta_over_u=[None if x is None else x / u for x in eta], omega_over_u=[None if x is None else x / u for x in om], solver_internal_final_eta=fnum(d['solver_final_eta_max']), solver_internal_final_omega=fnum(d['solver_final_omega_max']), zero_reference_components=d['ezero_count'] if use_exact else d['fzero_count'], evaluation_precision='FP64 (fma, sequential per row) from the stored inputs', nonfinite_reason=None if finite else ('skipped/runtime error' if (d['skipped'] or not d['ok']) else 'nonfinite entries in returned X or residual')),
                                       validation=dict(both_backward=bb, backward_strict_10u=('pass' if (bb == 'pass' and qual(eta, 10 * u) == 'pass' and qual(om, 10 * u) == 'pass') else ('fail' if (finite and not d['warmup']) else 'not_evaluated')), warmup_metrics_only=d['warmup'], reference_grade=grade if finf else 'unavailable', forward_10u=fq(10), forward_100u=fq(100), forward_reference_uncertainty_proxy=ref_unc, independent_evaluation_precision='double-double residual audit where present (external_validation.independent_audit)', b_integrity_ok=d.get('b_integrity_ok')),
                                       memory=dict(peak_device_bytes_tracked=d['peak_device_bytes_tracked'], device_free_after_bytes=d['device_free_after_bytes'], method_extra_bytes_tracked=d['method_extra_bytes_tracked'], irs_workspace_bytes=d['irs_workspace_bytes'] if d['irs_workspace_bytes'] else None, host_pinned_bytes=(case.get('memory_plan', {}).get('stored_matrix_bytes') if d['mode'] == 'host' else 0)),
                                       solution_path=sol, solution_sha256=xs, solution_dtype=d['x_dtype'] if sol else None, history_available=(d['method_id'].startswith(('native_', 'hetero', 'custom', 'policy', 'magma_ht'))), reuse=d.get('reuse', False)))
                    for _k in _per_rhs_keys:
                        if not trials[-1]['metrics'].get(_k): trials[-1]['metrics'].pop(_k, None)
                    key = (cid, d['method_id'], phase, run_kind)
                    cv = cov.setdefault(key, dict(planned=0, observed=0, reasons=set(), ok=0, finite=0, timeout=False))
                    cv['observed'] += 1; cv['ok'] += int(d['ok'] and not d['skipped']); cv['finite'] += int(finite)
                    if d['skipped']: cv['reasons'].add('memory_skip')
                    if not d['ok']: cv['reasons'].add('unsupported_vendor_nrhs' if 'status 21' in (d.get('error') or '') else 'runtime_error: ' + (d.get('error') or '')[:80])
                    if d['ok'] and not d['skipped'] and not d.get('metrics_finite', False): cv['reasons'].add('nonfinite_output')
            # planned counts
            for jb in jobs:
                j = jb['job']; base_kind = j.get('run_kind') or SUITE_KIND[suite]
                for mth in j['methods']:
                    for mode in j['modes'].split(','):
                        rk = 'timed_host' if mode == 'host' and base_kind in ('timed_resident', 'timed_host') else base_kind
                        cov.setdefault((cid, mth, j['phase'], rk), dict(planned=0, observed=0, reasons=set(), ok=0, finite=0, timeout=False))['planned'] += j['reps']
                        cov.setdefault((cid, mth, j['phase'], 'warmup'), dict(planned=0, observed=0, reasons=set(), ok=0, finite=0, timeout=False))['planned'] += j['warmup']
                        if jb['rc'] == -2: cov[(cid, mth, j['phase'], rk)]['reasons'].add('skipped_time_budget'); cov[(cid, mth, j['phase'], 'warmup')]['reasons'].add('skipped_time_budget')
                        elif jb['rc'] != 0: cov[(cid, mth, j['phase'], rk)]['reasons'].add('job_rc_%d%s' % (jb['rc'], '_timeout' if jb['rc'] == -9 else ''))
            # histories
            for hf in sorted(cdir.glob('hist_*.jsonl')):
                for line in open(hf):
                    h = json.loads(line); jb = tag2job.get(h['tag'], jobs[0]); j = jb['job']; base_kind = j.get('run_kind') or SUITE_KIND[suite]
                    run_kind = 'timed_host' if h['mode'] == 'host' and base_kind in ('timed_resident', 'timed_host') else base_kind
                    tid = trial_id(cid, h['method_id'], j['phase'], run_kind, h['rep'], h['tag'], h['mode'])
                    if h['kind'] == 'outer':
                        hists.append(dict(trial_id=tid, level='outer', counter='outer_correction_index', outer=h['outer'], stage=h['stage'], elapsed_s=h['t_s'], eta_max=fnum(h['eta_max']), omega_max=fnum(h['omega_max']), residual_inf_max=fnum(h['rinf_max']), inner_in_step=h['inner_in_step'], residual_kind='true (recomputed FP64 original-system residual)', precision='FP64', note=h['note'], transition=('escalate' in h['note'] or 'switch' in h['note'])))
                    else:
                        hists.append(dict(trial_id=tid, level='inner', counter='arnoldi_step', outer=h['outer'], cycle=h['cycle'], j=h['j'], rhs=h['rhs'], recursive_relative_residual=fnum(h['recursive_rel']), true_relative_residual=fnum(h['true_rel']) if h['true_rel'] is not None and h['true_rel'] >= 0 else None, residual_kind='recursive (Givens) and true at cycle end', note=h['note']))
    integrate_e1(cases, trials, hists, cov, src_sha, frozen_sha)
    # ---- write
    with open(OUT / 'cases.jsonl', 'w') as f:
        for c in cases.values(): f.write(json.dumps(c) + '\n')
    with open(OUT / 'trials.jsonl', 'w') as f:
        for t in trials: f.write(json.dumps(t) + '\n')
    with open(OUT / 'histories.jsonl', 'w') as f:
        for h in hists: f.write(json.dumps(h) + '\n')
    with open(OUT / 'coverage.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['case_id', 'method_id', 'phase', 'run_kind', 'planned_repetitions', 'observed_repetitions', 'coverage_status', 'reason'])
        for (cid, m, ph, rk), v in sorted(cov.items()):
            if v['observed'] == v['planned'] and v['finite'] == v['observed'] and not v['reasons']: st, reason = 'complete', ''
            elif v['observed'] == 0: st, reason = ('memory_skip' if 'memory_skip' in v['reasons'] else ('timeout' if any('timeout' in r for r in v['reasons']) else ('skipped_time_budget' if 'skipped_time_budget' in v['reasons'] else 'not_run'))), '; '.join(sorted(v['reasons'])) or 'job did not produce records (queue stopped or crashed)'
            else:
                r = sorted(v['reasons']); st = 'unsupported' if any('unsupported' in x for x in r) else ('memory_skip' if 'memory_skip' in r else ('numerical_failure' if 'nonfinite_output' in r else ('runtime_error' if any('runtime_error' in x for x in r) else ('partial' if v['observed'] < v['planned'] else 'complete_with_notes'))))
                reason = '; '.join(r) if r else (f"observed {v['observed']} of planned {v['planned']}" )
            w.writerow([cid, m, ph, rk, v['planned'], v['observed'], st, reason])
    json.dump(CATALOG, open(OUT / 'methods.json', 'w'), indent=1)
    print(f'cases={len(cases)} trials={len(trials)} histories={len(hists)} coverage_rows={len(cov)} source_sha={src_sha[:12]}')
    json.dump(dict(source_sha256=src_sha, frozen_sha256=frozen_sha, n_cases=len(cases), n_trials=len(trials)), open(OUT / 'summaries' / 'collect_stats.json', 'w'))


# ---------------------------------------------------------------- E1 scalar arithmetic-model track integration (sub-track output under e1_scalar/out)
def integrate_e1(cases, trials, hists, cov, src_sha, frozen_sha):
    E1 = ROOT / 'e1_scalar' / 'out'
    if not (E1 / 'trials.jsonl').exists(): return
    for d in ['matrices/e1', 'rhs/e1', 'solutions/e1', 'references/e1']: (OUT / d).mkdir(parents=True, exist_ok=True)
    e1cfg = hashlib.sha256(open(E1 / 'run_meta.json', 'rb').read()).hexdigest() if (E1 / 'run_meta.json').exists() else None
    for line in open(E1 / 'cases.jsonl'):
        c = json.loads(line); cid = 'e1_' + c['case_id']
        shutil.copyfile(E1 / c['a_path'], OUT / 'matrices' / 'e1' / Path(c['a_path']).name); shutil.copyfile(E1 / c['b_path'], OUT / 'rhs' / 'e1' / Path(c['b_path']).name)
        for k in ('reference_path', 'reference_txt_path'):
            if c.get(k) and (E1 / c[k]).exists(): shutil.copyfile(E1 / c[k], OUT / 'references' / 'e1' / Path(c[k]).name)
        cases[cid] = dict(case_id=cid, suite='E1', n=c['n'], nrhs=1, family=c['family'], spectrum='geometric', seed=c['seed'], dtype='fp32', rhs='random', kappa_nominal=c['kappa_nominal'],
                          conditioning_measured=dict(kappa2=c.get('kappa2_measured'), kappa2_method=c.get('kappa2_measured_method'), kappa_inf=c.get('kappa_inf_measured'), kappa_inf_method=c.get('kappa_inf_measured_method')),
                          a_sha256=c['a_sha256'], b_sha256=c['b_sha256'], hash_order=c.get('sha256_convention'), generator=dict(name='tpir.problems.make_problem (prior three_precision_lab, Haar-up-to-signs QR orientation)', version=src_sha, description=c.get('generator')), rhs_construction=c.get('rhs_construction'),
                          inputs=dict(mode='embedded', a_path=f"matrices/e1/{Path(c['a_path']).name}", b_path=f"rhs/e1/{Path(c['b_path']).name}", format='numpy .npy float32, C order'),
                          reference=dict(grade=c.get('reference_grade'), method=c.get('reference_method'), digits=c.get('reference_digits'), audit_digits=c.get('audit_digits'), audit_maxrel=c.get('reference_audit_maxrel'), residual_inf_mp100=c.get('reference_residual_inf_mp100'), numerical_reference_path=f"references/e1/{Path(c['reference_path']).name}" if c.get('reference_path') else None),
                          half_factor_sha256=c.get('half_factor_sha256'), memory_plan=dict(stored_matrix_bytes=c['n'] * c['n'] * 4), target='fp32', tau=100 * U32, e1_source_case=c)
    e1m = json.load(open(E1 / 'methods.json'))
    for m in e1m:
        m = dict(m); m['backend'] = 'CPU scalar operation-rounded arithmetic model (prior three_precision_lab tpir package, numpy 2.5.4, Python 3.14.7); NOT a performance implementation'; m['display_name'] = m.get('display_name') or m['method_id']; m['algorithm'] = m.get('algorithm') or 'LU with partial pivoting + iterative refinement (scalar model)'
        m['precision_source'] = 'source_audited (explicit operation-rounded arithmetic)'; m['performance_claims_permitted'] = False
        if not any(x['method_id'] == m['method_id'] for x in CATALOG): CATALOG.append(m)
    for line in open(E1 / 'trials.jsonl'):
        t = json.loads(line); cid = 'e1_' + t['case_id']; m = t['metrics']; q = t['qualification']
        xdst = OUT / 'solutions' / 'e1' / Path(t['x_path']).name; shutil.copyfile(E1 / t['x_path'], xdst)
        hp = E1 / t['history_path']; tid = 'e1__' + t['trial_id']
        if hp.exists():
            try:
                H = json.load(open(hp)); rows = H if isinstance(H, list) else H.get('history', H.get('outer', []))
                for i, h in enumerate(rows if isinstance(rows, list) else []):
                    if isinstance(h, dict): hists.append(dict(trial_id=tid, level='outer', counter='correction_index', outer=h.get('correction', h.get('outer', i)), elapsed_s=h.get('elapsed_s'), eta=h.get('eta'), omega=h.get('omega'), finf=h.get('finf'), update_norm=h.get('update_norm'), stage=h.get('stage'), residual_kind='true (FP64 stored-system residual)', raw=h))
            except Exception as e: hists.append(dict(trial_id=tid, level='outer', note=f'history unreadable: {e}'))
        finite = not t.get('nonfinite', False) and m.get('eta') is not None
        def q1(v, tau): return 'not_evaluated' if v is None else ('pass' if v <= tau else 'fail')
        bb = 'pass' if (finite and q['eta_le_100u'] and q['omega_le_100u']) else ('fail' if m.get('eta') is not None else 'not_evaluated')
        trials.append(dict(trial_id=tid, case_id=cid, method_id=t['method_id'], suite='E1', phase='evaluation', rep=0, run_kind='diagnostic', mode='cpu_scalar_model', tag='E1', source_sha256=src_sha, config_sha256=frozen_sha, e1_config_sha256=e1cfg, timestamp=None, method_order_seed=None, environment_id='env/capabilities.json',
                           target_profile=dict(id='target32_primary_100u', dtype='fp32', tau_normwise=100 * U32, tau_componentwise=100 * U32, u=U32, strict_tau=10 * U32), target_precision='fp32', working_precision='fp32',
                           terminal_reason=t['terminal_reason'], external_validation=dict(evaluator='E1 driver FP64 metrics from exact stored FP32 inputs; final residual also at 100 digits (residual_inf_mp100)'), actual_final_stage=t['method_id'], used_fallback=False, skipped=False, ok=True, error=t.get('failure'),
                           timings_s=dict(host_e2e=None, device_ready_wall=None, scalar_model_cpu_seconds=t.get('cpu_seconds')), timing_boundary='not applicable (CPU scalar arithmetic model; cpu seconds are not a performance measurement)', warmup=False, profiled=False,
                           counters=dict(outer=t['outer_iters'], inner_total=t['inner_iters_total'], matvecs=t.get('matvecs'), preconditioner_applications=t.get('precond_applies'), first_attainment_10u=t.get('first_attainment_10u'), gmres_params=t.get('gmres_params')),
                           metrics=dict(normwise_backward_per_rhs=[m.get('eta')], componentwise_backward_per_rhs=[m.get('omega')], forward_inf_per_rhs=[m.get('finf')], forward_2_per_rhs=[m.get('f2')], forward_componentwise_nonzero_per_rhs=[m.get('fcomp_nonzero')], relative_residual_inf_per_rhs=[m.get('relres_inf')], relative_residual_2_per_rhs=[m.get('relres_2')], residual_inf_mp100=m.get('residual_inf_mp100'), evaluation_precision='FP64 from exact stored FP32 A,b; 100-digit final residual', nonfinite_reason=t.get('nonfinite_reason')),
                           validation=dict(both_backward=bb, backward_strict_10u='pass' if (finite and q['eta_le_10u'] and q['omega_le_10u']) else ('fail' if finite else 'not_evaluated'), reference_grade='high_precision_verified', forward_10u=q1(m.get('finf'), 10 * U32), forward_100u=q1(m.get('finf'), 100 * U32), factor_sha256=t.get('factor_sha256'), factor_hash_verified=t.get('factor_hash_verified_against_shared_factorization')),
                           memory=dict(note='host numpy arrays; not measured'), solution_path=f"solutions/e1/{xdst.name}", solution_sha256=t['x_sha256'], solution_dtype='float32 (.npy)', history_available=True, precision_contract=t.get('precision_contract')))
        cov[(cid, t['method_id'], 'evaluation', 'diagnostic')] = dict(planned=1, observed=1, reasons=set(), ok=1, finite=1 if finite else 0, timeout=False)
        if not finite: cov[(cid, t['method_id'], 'evaluation', 'diagnostic')]['reasons'].add('nonfinite_output')

if __name__ == '__main__': main()
