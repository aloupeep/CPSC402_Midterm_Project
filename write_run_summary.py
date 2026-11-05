#!/usr/bin/env python3
"""Generate RUN_SUMMARY.md (factual, observation-only) from the collected files and summaries."""
import csv, json, statistics as st, time
from collections import defaultdict, Counter
from pathlib import Path
OUT = Path('/home/worker/lab/campaign/writeup_results'); S = OUT / 'summaries'
cases = {json.loads(l)['case_id']: json.loads(l) for l in open(OUT / 'cases.jsonl')}
trials = [json.loads(l) for l in open(OUT / 'trials.jsonl')]
stats = json.load(open(S / 'collect_stats.json'))
nw = [t for t in trials if t['run_kind'] != 'warmup']
by_suite = Counter(t['suite'] for t in nw); by_kind = Counter(t['run_kind'] for t in trials)
term = Counter(t['terminal_reason'] for t in nw)
cov = list(csv.DictReader(open(OUT / 'coverage.csv'))); covst = Counter(r['coverage_status'] for r in cov)
def rows(name): return list(csv.DictReader(open(S / name))) if (S / name).exists() else []
best = defaultdict(lambda: [0, 0])
for t in trials:
    if t['run_kind'] not in ('timed_resident', 'timed_host') or t['suite'] != 'E4': continue
    n = cases[t['case_id']]['n']; k = (t['target_precision'], t['method_id']); best[k][0] = max(best[k][0], n)
    if t['validation']['both_backward'] == 'pass': best[k][1] = max(best[k][1], n)
def md_table(hdr, rws):
    return '| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(str(x) for x in r) + ' |' for r in rws) + '\n'
L = []
L.append(f"# RUN_SUMMARY — mixed-precision LU and iterative refinement campaign (Tesla V100-SXM2-32GB, {time.strftime('%Y-%m-%d')})\n")
L.append("Observation-only summary generated from `trials.jsonl`, `cases.jsonl`, `coverage.csv` and `summaries/*.csv`. Interpretation is left to the independent analysis. All times are seconds; 'device-ready' = inputs resident, 'host_e2e' = host-to-host.\n")
L.append("## 1. What ran versus the plan\n")
L.append(f"- Records: {len(cases)} cases, {len(trials)} trial records ({len(nw)} non-warmup; warmup records are kept with `run_kind=warmup`). Suites (non-warmup records): {dict(sorted(by_suite.items()))}. Run kinds: {dict(sorted(by_kind.items()))}.\n")
L.append(f"- Coverage statuses over (case, method, phase, run_kind) groups: {dict(sorted(covst.items()))}. Every planned-but-unobserved combination has a reason in `coverage.csv`.\n")
L.append(f"- Terminal reasons (non-warmup): {dict(term.most_common())}.\n")
L.append("- Planned but not run / reduced (see LIMITATIONS.md): E1 n=512; KKT family; nrhs=8 above n=8192 except n=16384 geometric seed 9200; seeds reduced to 2 at n=16384 and 1 at n>=12288 (other sizes); at n>=32768 only the core method group (plus the cuSOLVER-IRS and MAGMA groups at 32768 geometric); 36864 (FP64, 10.1 GiB) and 49152 (FP32, 9.06 GiB) for the core group only; repetitions 7/5/3 (n<=8192 / 16384 / >=24576) and host-mode repetitions 2/1.\n")
L.append("- Development (seeds 9100/9101, `phase=development`): block size (128/256/512), panel implementation, GMRES restart/tolerance grid, baseline choice; full table in `summaries/dev_tuning.csv` and the dev records in trials.jsonl. Frozen settings and their amendment history: `configs/frozen_config.json`, `configs/FROZEN_CONFIG_HISTORY.md`.\n")
L.append("## 2. Largest n per method (E4 timed records; second number = largest n with a primary-profile qualified record)\n")
L.append(md_table(['target', 'method', 'largest n run', 'largest n qualified'], [(k[0], k[1], v[0], v[1]) for k, v in sorted(best.items())]))
L.append("## 3. Backends\n- Ordinary baselines: cuSOLVER 11.7.1 `cusolverDnXgetrf/Xgetrs` FP64 (`native_lu64`, frozen main denominator for target64) and FP32 (`native_lu32`, target32); MAGMA 2.9.0 native (`magma_lu64_native`, `magma_lu32_native`) and hybrid (`magma_lu64_hybrid`, development only) LU.\n- Mature compiled mixed endpoints: cuSOLVER IRS `cusolverDnIRSXgesv` (FP64/FP16, FP64/FP32, FP32/FP16; classical and GMRES modes) and MAGMA `magma_dxgesv_gmres_gpu` (SHT = FP16-tensor-core factorization, SS = FP32; IRSTRS classical, GMSTRS GMRES) plus `magma_htgetrf_gpu` factors inside this campaign's refinement engine.\n- Custom compiled: heterogeneous FP32/FP16-TC blocked LU (`hetero16_*`), FP32 same-code control (`custom32_*`), stationary IR, FGMRES-IR, left GMRES-IR, recovery policies — all device-resident (no per-iteration host vectors).\n- Scalar operation-rounded model (E1, CPU): hss/hsd/sss/ssd, input16_ssd, half_factors_solve32, paper-aligned left GMRES-IR (app64/app32); no performance claims.\n")
L.append("## 4. Optimization gates and profile evidence\n")
dev = rows('dev_tuning.csv')
L.append("- Development n=8192 (seed 9100, FP64 inputs, geometric κ=1e2): cuSOLVER Xgetrf FP32 factor 0.0635 s, FP64 0.102 s; custom FP32 blocked LU 0.098 s (1.55× the FP32 vendor factorization: **fails** the project's 1.5× promotion gate narrowly; 1.44× at n=4096); heterogeneous FP16-TC LU 0.082 s (nb=512). The custom implementation is panel-bound (tall-panel cuSOLVER calls); the MAGMA native recursive panel gave 8% and was not adopted. Interpretation of `hetero16_*` timings is restricted accordingly; the vendor mixed drivers are the mature endpoints.\n")
L.append("- GMRES development grid (κ=1e6, n=4096): FP32-factor GMRES-IR converged with 8 inner iterations for every (m, tol) in {20,40,80}×{1e-2,1e-4,1e-6}; FP16-TC-factor GMRES-IR failed for every setting (budget 480). Frozen: m=40, tol=1e-4, budget 480.\n")
L.append("- Profiles: instrumented stage timers and counters for n=8192 and n=32768 plus a hard GMRES case (`profiles/`, `summaries/profiles_stage_breakdown.csv`, `figures/e8_stage_breakdown.png`). No nsys traces (not installed).\n")
L.append("## 5. Accuracy qualification by suite / target / family (primary profile: η and ω ≤ 100·u_target on the original stored system; strict 10·u; forward vs reference)\n")
q = rows('qualification_by_suite_target_family_method.csv')
L.append(md_table(['suite', 'target', 'family', 'method', 'records', 'pass 100u', 'pass 10u', 'fwd eval', 'fwd pass 10u', 'fwd pass 100u', 'nonfinite', 'unsupported', 'mem skip', 'fallback'], [(r['suite'], r['target'], r['family'], r['method_id'], r['trials'], r['pass_primary_100u'], r['pass_strict_10u'], r['forward_evaluated'], r['forward_pass_10u'], r['forward_pass_100u'], r['nonfinite'], r['unsupported'], r['memory_skip'], r['used_fallback']) for r in q]))
L.append("## 6. E4 timing ratios (descriptive). Two denominators: the frozen ordinary LU (`native_lu64`/`native_lu32`; its own qualification at 100u is reported per case — it FAILED the componentwise criterion on most n>=8192 cases) on cases where the method qualified, and the qualified full-precision-refinement baseline (`native_ir64_w64_r64`/`native_ir32_w32_r64`) on cases where both qualified. Every per-case ratio: `summaries/e4_paired_speedups.csv`; raw medians/min/max/IQR: `summaries/e4_timing_per_case_method.csv`.\n")
sp = rows('e4_speedup_summary.csv')
L.append(md_table(['target', 'n', 'nrhs', 'method', 'method-qualified / total', 'median T(ordinary LU)/T(method)', 'min', 'max', 'both-qualified vs refined', 'median T(refined baseline)/T(method)', 'min', 'max'], [(r['target'], r['n'], r['nrhs'], r['method_id'], f"{r['method_qualified_cases']}/{r['total_cases']}", r['median_ratio_vs_ordinary_lu'], r['min_ratio_vs_ordinary_lu'], r['max_ratio_vs_ordinary_lu'], r['both_qualified_cases_vs_refined'], r['median_ratio_vs_refined_baseline'], r['min_ratio_vs_refined_baseline'], r['max_ratio_vs_refined_baseline']) for r in sp]))
tm = rows('e4_timing_per_case_method.csv')
L.append("### Device-ready medians, nrhs=1, geometric κ=1e2, seed 9200 (all methods, qualified flag)\n")
L.append(md_table(['target', 'n', 'method', 'device-ready median s', 'host_e2e median s', 'qualified all reps', 'outer', 'inner', 'terminal'], [(r['target'], r['n'], r['method_id'], r['device_ready_median_s'], r['host_e2e_median_s'], r['qualified_all_reps'], r['median_outer'], r['median_inner'], r['terminal_reasons']) for r in tm if r['nrhs'] == '1' and r['spectrum'] == 'geometric' and r['seed'] == '9200']))
L.append("## 7. E6 recovery policies (target64; pass = primary profile; stage = final factorization precision)\n")
L.append(md_table(['phase', 'n', 'case family', 'policy', 'pass/cases', 'median time s', 'max time s', 'final stages', 'median inner', 'median refactorizations'], [(r['phase'], r['n'], r['case_family'], r['method_id'], f"{r['pass_primary']}/{r['cases']}", r['median_device_ready_s'], r['max_device_ready_s'], r['final_stages'], r['median_inner'], r['median_refactorizations']) for r in rows('e6_policies.csv')]))
L.append("## 8. E3 scaling / range (FP64 inputs, geometric κ=1e4)\n")
L.append(md_table(['n', 'global 2^s', 'row/col span', 'method', 'pass/seeds', 'nonfinite', 'median outer', 'fallback'], [(r['n'], r['global_exp'], r['rowcol_span'], r['method_id'], f"{r['pass_primary']}/{r['seeds']}", r['nonfinite'], r['median_outer'], r['fallback']) for r in rows('e3_scaling.csv')]))
L.append("## 9. E5 factor reuse (geometric κ=1e2, FP64 inputs; reuse = solve+refinement with reused factors, fresh = factor+solve)\n")
L.append(md_table(['n', 'nrhs', 'method', 'reuse solve median s', 'fresh solve median s', 'factor median s', 'per-RHS reuse s', 'qualified/records'], [(r['n'], r['nrhs'], r['method_id'], r['reuse_solve_median_s'], r['fresh_solve_median_s'], r['factor_median_s'], r['per_rhs_reuse_s'], f"{r['qualified']}/{r['records']}") for r in rows('e5_reuse.csv')]))
L.append("## 10. E1 scalar three-precision model (CPU, 100-digit references; final iterate)\n")
L.append(md_table(['method', 'nominal κ', 'cases', 'backward pass 100u32', 'forward pass 10u32', 'forward pass 100u32', 'nonfinite'], [(r['method_id'], r['kappa_nominal'], r['cases'], r['backward_pass_100u'], r['forward_pass_10u'], r['forward_pass_100u'], r['nonfinite']) for r in rows('e1_scalar_qualification.csv')]))
L.append("## 11. References, audits and failures\n")
aud = (OUT / 'tests' / 'AUDIT_SUMMARY.md').read_text() if (OUT / 'tests' / 'AUDIT_SUMMARY.md').exists() else 'audit summary missing'
L.append(aud.replace('# Independent audit summary (generated by audit_summary.py)', '').strip() + '\n')
L.append("- Forward references: `exact_stored` for the dyadic family (sign solutions, unit-column solutions), `numerical_reference` (FP64/FP32 LU + double-double residual refinement, uncertainty proxy recorded) for the other compiled-track families, `high_precision_verified` (mpmath 100 digits, 120-digit recheck) for E1. Forward qualification is marked `indeterminate` when the reference uncertainty proxy exceeds 10% of the threshold.\n")
L.append(f"- Numerical failures and vendor anomalies are kept as records: nonfinite outputs {sum('nonfinite' in (t['terminal_reason'] or '') for t in nw)}, unsupported vendor combinations {sum('unsupported' in (t['terminal_reason'] or '') for t in nw)}, runtime errors {sum((not t['ok']) and 'unsupported' not in (t['terminal_reason'] or '') for t in nw)}, memory skips {sum(t['skipped'] for t in nw)}.\n")
L.append("- E0 regression: the prior teaching case (n=64, nominal κ=1e3) saved iterates were replayed against independent residual evaluation (tests/e0_teaching_case_replay/): recorded prior forward errors reproduced (hsd 3.0261e-8, hss 1.6535e-5); prior scalar package tests 39/39 pass; handoff tool tests 18/18 pass.\n")
L.append("## 12. Reproduction\n```\n# build (CUDA 12.6, sm_70, MAGMA 2.9.0 under campaign/build_magma):\nbash build/build.sh\n# plan + run suites (single GPU owner):\npython3 campaign.py plan --suite dev,E2,E6,E4,E2a,E3,E5,E8 --sizes64 8192,12288,16384,24576,32768,36864 --sizes32 8192,16384,32768,49152\npython3 campaign.py run --suite E2,E6,E4,E2a,E3,E5,E8\n# audit, collect, tables/figures, package:\npython audit.py; python3 collect.py; python summarize.py; python3 audit_summary.py; python3 write_run_summary.py; python3 finalize.py\npython tools/validate_results.py writeup_results && python tools/package_results.py writeup_results --out mixed_precision_writeup_results.zip\n```\nA single case can be regenerated and re-audited with the command stored in `matrices/recipes/<case_id>.json` (`regen_command`).\n")
(OUT / 'RUN_SUMMARY.md').write_text('\n'.join(L)); print('RUN_SUMMARY.md written', len('\n'.join(L)), 'chars')
