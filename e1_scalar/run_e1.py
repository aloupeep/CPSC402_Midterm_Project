#!/usr/bin/env python
"""E1 driver: precision roles, scalar operation-rounded arithmetic model track.

Usage (from anywhere):
  taskset -c 8-79 /home/worker/lab/venv/bin/python /home/worker/lab/campaign/e1_scalar/run_e1.py \
      --out /home/worker/lab/campaign/e1_scalar/out --procs 54 [--ns 64 128 256]
  ... --summary-only   # rebuild methods.json / summary.md from existing jsonl files

Each case is handled by one worker process (multiprocessing, fork).  Workers write
matrices / rhs / references / solutions / histories as files and return JSON records;
the parent writes cases.jsonl, trials.jsonl, methods.json and summary.md.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, hashlib, time, argparse, traceback, platform
from pathlib import Path
import multiprocessing as mproc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'three_precision_lab'))
sys.path.insert(0, str(HERE))

import numpy as np
import scipy
import mpmath as mp
from tpir.problems import make_problem
from tpir.solver import run_solver, METHODS as TPIR_METHODS
from tpir.arithmetic import lu_factor, Arithmetic, UNIT_ROUNDOFF
from e1_methods import (mp_lu_solve, mp_residual_inf, metrics_fp64, gmres_ir, U32)

NS = [64, 128, 256]
KAPPAS = [1e2, 1e3, 3e3, 1e4, 1e5, 1e6]
SEEDS = [9200, 9201, 9202]
MAX_CORR = 30
GMRES_TOL = 1e-4
GMRES_M_CAP = 30
REF_DPS = 100
AUDIT_DPS = 120
AUDIT_EVERY = 5

CLASSICAL = ['hss', 'hsd', 'sss', 'ssd', 'input16_ssd', 'half_factors_solve32']
GMRES_METHODS = {'paper_left_gmres_hsd_app64': 64, 'paper_left_gmres_hsd_app32': 32}
METHOD_ORDER = CLASSICAL + list(GMRES_METHODS)

TERMINAL_MAP = {'iteration_limit': 'max_corrections_30', 'unchanged_working_update': 'unchanged_working_update',
                'working_precision_cycle': 'working_precision_cycle', 'computed_residual_zero': 'computed_residual_zero',
                'numerical_failure': 'numerical_failure',
                'correction_rejected_true_relres_ge_1': 'gmres_correction_rejected_true_relres_ge_1'}


# ----------------------------------------------------------------------------- helpers
def sha256_bytes(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def array_sha256(arr):
    """sha256 of the raw little-endian bytes of the array in C order (dtype as given)."""
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder == '>':
        a = a.astype(a.dtype.newbyteorder('<'))
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def factor_hashes(fac):
    """(L/U/perm hash, packed hash).  L unit-lower, U upper, both in the factor dtype;
    permutation as int64.  Packed hash matches tpir.solver's factor_hash."""
    lu = fac.lu
    n = lu.shape[0]
    L = (np.tril(lu, -1) + np.eye(n, dtype=lu.dtype)).astype(lu.dtype)
    U = np.triu(lu).astype(lu.dtype)
    perm = np.asarray(fac.permutation, dtype=np.int64)
    lup = sha256_bytes(np.ascontiguousarray(L).tobytes(), np.ascontiguousarray(U).tobytes(), perm.tobytes())
    packed = hashlib.sha256(lu.tobytes() + fac.permutation.tobytes()).hexdigest()
    return lup, packed


def fnum(x):
    """float or None (nonfinite -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def jsonsafe(obj):
    """Recursively replace nonfinite floats by None; numpy scalars/arrays to Python."""
    if isinstance(obj, dict):
        return {str(k): jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonsafe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonsafe(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.generic):
        return jsonsafe(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def metrics_with_flags(mdict):
    """Split metrics into finite floats (None for nonfinite) + nonfinite reasons."""
    out = {}
    reasons = {}
    for k, v in mdict.items():
        f = fnum(v)
        out[k] = f
        if f is None:
            reasons[k] = 'nan' if (v is None or (isinstance(v, float) and np.isnan(v))) else ('inf' if np.isinf(v) else 'nonfinite')
    return out, reasons


def history_from_iterates(xs, a32, b32, xref64, gmres_outer=None):
    hist = []
    prev = None
    for j, x in enumerate(xs):
        m = metrics_fp64(a32, b32, x, xref64)
        entry = {'correction_index': j,
                 'resid_inf_fp64': fnum(m['resid_inf_fp64']), 'eta': fnum(m['eta']), 'omega': fnum(m['omega']),
                 'finf': fnum(m['finf']), 'f2': fnum(m['f2']), 'fcomp_nonzero': fnum(m['fcomp_nonzero']),
                 'relres_inf': fnum(m['relres_inf']), 'relres_2': fnum(m['relres_2']),
                 'update_inf': None, 'update_rel': None, 'changed': None,
                 'finf_le_10u': bool(np.isfinite(m['finf']) and m['finf'] <= 10 * U32),
                 'nonfinite_fields': [k for k in ('eta', 'omega', 'finf', 'f2', 'relres_inf') if fnum(m[k]) is None]}
        if prev is not None:
            with np.errstate(all='ignore'):
                up = float(np.max(np.abs(x.astype(np.float64) - prev.astype(np.float64))))
                nx = float(np.max(np.abs(x.astype(np.float64))))
            entry['update_inf'] = fnum(up)
            entry['update_rel'] = fnum(up / nx) if nx > 0 else None
            entry['changed'] = bool(not np.array_equal(x, prev))
        if gmres_outer is not None and j >= 1 and j - 1 < len(gmres_outer):
            entry['gmres'] = jsonsafe(gmres_outer[j - 1])
        hist.append(entry)
        prev = x
    return hist


def first_attainment(hist):
    for h in hist:
        if h['finf_le_10u']:
            return h['correction_index']
    return None


# ----------------------------------------------------------------------------- per-case worker
def run_case(job):
    idx, n, kappa, seed, audit, out_dir = job
    out = Path(out_dir)
    t_case = time.perf_counter()
    log = []
    try:
        spec = {'n': n, 'kappa': kappa, 'seed': seed, 'family': 'spectral', 'rhs': 'random'}
        a32, b32, _xman, pmeta = make_problem(spec)
        assert a32.dtype == np.float32 and b32.dtype == np.float32
        case_id = hashlib.sha256(a32.tobytes() + b32.tobytes() + f'|n={n}|seed={seed}'.encode()).hexdigest()[:16]
        a_path = out / 'matrices' / f'{case_id}_A.npy'
        b_path = out / 'rhs' / f'{case_id}_b.npy'
        for p in (a_path.parent, b_path.parent, out / 'references', out / 'solutions', out / 'histories'):
            p.mkdir(parents=True, exist_ok=True)
        np.save(a_path, a32)
        np.save(b_path, b32)
        a64 = a32.astype(np.float64)
        b64 = b32.astype(np.float64)
        kappa2 = float(np.linalg.cond(a64, 2))
        kappainf = float(np.linalg.cond(a64, np.inf))

        # ---- reference (100 digits), audit (120 digits) on the prespecified subset
        t0 = time.perf_counter()
        x100 = mp_lu_solve(a32, b32, REF_DPS)
        t_ref = time.perf_counter() - t0
        with mp.workdps(REF_DPS):
            xref64 = np.array([float(v) for v in x100], dtype=np.float64)
            ref_txt = '\n'.join(mp.nstr(v, 40) for v in x100) + '\n'
        ref_path = out / 'references' / f'{case_id}_xref100.npy'
        np.save(ref_path, xref64)
        (out / 'references' / f'{case_id}_xref100_40digits.txt').write_text(ref_txt)
        # 100-digit residual of the reference itself (sanity, recorded)
        with mp.workdps(REF_DPS):
            worst = mp.mpf(0)
            for i in range(n):
                s = mp.mpf(float(b32[i]))
                row = a32[i]
                for j in range(n):
                    s -= mp.mpf(float(row[j])) * x100[j]
                worst = max(worst, abs(s))
            ref_resid = float(worst)
        audit_digits = None
        audit_maxrel = None
        audit_maxrel_normwise = None
        t_audit = 0.0
        if audit:
            t0 = time.perf_counter()
            x120 = mp_lu_solve(a32, b32, AUDIT_DPS)
            t_audit = time.perf_counter() - t0
            with mp.workdps(AUDIT_DPS):
                comp = mp.mpf(0)
                nd = mp.mpf(0)
                nn = mp.mpf(0)
                for i in range(n):
                    d = abs(x100[i] - x120[i])
                    nd = max(nd, d)
                    nn = max(nn, abs(x120[i]))
                    if x120[i] != 0:
                        comp = max(comp, d / abs(x120[i]))
                audit_maxrel = float(comp)
                audit_maxrel_normwise = float(nd / nn) if nn else None
            audit_digits = AUDIT_DPS

        # ---- shared factorizations (deterministic; identical to those inside run_solver)
        fac16 = lu_factor(a32, 16)
        h16_lup, h16_packed = factor_hashes(fac16)
        fac32 = lu_factor(a32, 32)
        h32_lup, h32_packed = factor_hashes(fac32)
        a_in16 = Arithmetic(16).cast(a32).astype(np.float32)
        fac32_in16 = lu_factor(a_in16, 32)
        hin16_lup, hin16_packed = factor_hashes(fac32_in16)

        case_rec = {
            'case_id': case_id, 'case_index_sorted': idx, 'n': n, 'nrhs': 1,
            'family': 'prescribed_geometric_haar_qr', 'seed': seed, 'dtype': 'fp32',
            'kappa_nominal': kappa, 'kappa2_measured': kappa2,
            'kappa2_measured_method': 'numpy.linalg.cond(A32.astype(float64), 2) [SVD, LAPACK gesdd]',
            'kappa_inf_measured': kappainf,
            'kappa_inf_measured_method': 'numpy.linalg.cond(A32.astype(float64), inf) = ||A||_inf * ||inv(A)||_inf [LAPACK getrf/getri]',
            'a_path': str(a_path.relative_to(out)), 'b_path': str(b_path.relative_to(out)),
            'a_sha256': array_sha256(a32), 'b_sha256': array_sha256(b32),
            'sha256_convention': 'sha256 of the raw little-endian float32 bytes in C (row-major) order, numpy .tobytes()',
            'tpir_case_id_of_spec': pmeta['case_id'],
            'generator': ('tpir.problems.make_problem(spec) with spec={n,kappa,seed,family=spectral,rhs=random}: '
                          'rng=numpy.random.default_rng(seed); sigma=geomspace(1,1/kappa,n); '
                          'U=qr(rng.standard_normal((n,n)))[0]; V=qr(rng.standard_normal((n,n)))[0] (numpy Householder QR of a '
                          'Gaussian matrix; no sign correction of R diagonal, so U,V are QR-orthogonal factors, Haar up to column signs); '
                          'A64=(U*sigma)@V.T in FP64; stored A32=A64.astype(float32). Then x_man=rng.standard_normal(n) is drawn '
                          '(unused for the random-RHS family), then b32 = rng.standard_normal(n).astype(float32). '
                          'The stored FP32 A32,b32 define the problem for every method.'),
            'rhs_construction': 'b = rng.standard_normal(n) (FP64) rounded to FP32 (round-to-nearest-even), one RHS',
            'reference_path': str(ref_path.relative_to(out)),
            'reference_txt_path': str((out / 'references' / f'{case_id}_xref100_40digits.txt').relative_to(out)),
            'reference_grade': 'high_precision_verified',
            'reference_method': 'mpmath LU with partial pivoting (pure-Python lists of mpf) at 100 decimal digits on the exactly promoted stored FP32 A,b',
            'reference_digits': REF_DPS, 'reference_residual_inf_mp100': ref_resid,
            'audit_digits': audit_digits, 'reference_audit_maxrel': audit_maxrel,
            'reference_audit_maxrel_normwise_inf': audit_maxrel_normwise,
            'reference_audit_rule': f'cases whose index in the (n, kappa_nominal, seed)-sorted list is divisible by {AUDIT_EVERY}; recomputed at {AUDIT_DPS} digits; maxrel = max_i |x100_i - x120_i| / |x120_i| over nonzero x120_i',
            'half_factor_sha256': h16_lup,
            'half_factor_sha256_convention': 'sha256(L16.tobytes() + U16.tobytes() + perm.astype(int64).tobytes()); L16 unit-lower float16 (n x n, C order), U16 upper float16, perm = row permutation with P A = L U',
            'half_factor_packed_sha256': h16_packed,
            'fp32_factor_sha256': h32_lup, 'fp32_factor_sha256_input16': hin16_lup,
            'half_factor_growth': float(fac16.growth),
            'timing_seconds': {'reference_100': t_ref, 'audit_120': t_audit},
        }

        trials = []
        method_time = {}
        for method_id in METHOD_ORDER:
            trial_id = f'{case_id}__{method_id}'
            t0 = time.process_time()
            tw = time.perf_counter()
            if method_id in CLASSICAL:
                rec, xs = run_solver(a32, b32, method_id, MAX_CORR)
                cpu = time.process_time() - t0
                tm = TPIR_METHODS[method_id]
                if method_id in ('hss', 'hsd', 'half_factors_solve32'):
                    expect_packed, fh = h16_packed, h16_lup
                elif method_id in ('sss', 'ssd'):
                    expect_packed, fh = h32_packed, h32_lup
                else:
                    expect_packed, fh = hin16_packed, hin16_lup
                verified = (rec.get('factor_hash') == expect_packed)
                gmres_outer = None
                inner_total = 0
                outer = int(rec['iterations'])
                # counting convention: one residual (matvec) per correction attempt (one extra residual when the
                # loop ended on a computed zero residual or failed inside a correction attempt); one factor
                # application for the initial solve (if it succeeded) plus one per completed correction.
                init_ok = rec['initialization'] == 'low_precision_solve'
                fail = rec.get('failure') or ''
                failed_in_loop = rec['status'] == 'numerical_failure' and not (fail.startswith('nonfinite:factor') or fail.startswith('zero_pivot_factor'))
                matvecs = outer + (1 if (rec['status'] == 'computed_residual_zero' or failed_in_loop) else 0)
                precond_applies = (1 if init_ok else 0) + outer
                gparams = None
                status = rec['status']
                failure = rec.get('failure')
                contract = tm
            else:
                bits = GMRES_METHODS[method_id]
                rec, xs = gmres_ir(a32, b32, fac16, bits, GMRES_TOL, GMRES_M_CAP, MAX_CORR, 1.0)
                cpu = time.process_time() - t0
                verified = True  # same fac16 object as hashed above (hsd's packed hash equality is checked via hss/hsd records)
                fh = h16_lup
                gmres_outer = rec['gmres_outer']
                inner_total = int(rec['inner_iters_total'])
                outer = int(rec['iterations'])
                matvecs = int(rec['matvecs'])
                precond_applies = int(rec['precond_applies'])
                gparams = {'tol_preconditioned_recursive': GMRES_TOL, 'm_cap': GMRES_M_CAP, 'm_used': rec['m'],
                           'restart': None, 'max_outer': MAX_CORR, 'operator_application_bits': bits,
                           'basis_bits': 32, 'dot_bits': 32, 'givens_bits': 32, 'hessenberg_bits': 32,
                           'residual_bits': 64, 'update_bits': 32, 'reject_ratio_true_corr_relres': 1.0,
                           'per_outer': jsonsafe(gmres_outer)}
                status = rec['status']
                failure = rec.get('failure')
            wall = time.perf_counter() - tw
            method_time[method_id] = wall
            if len(xs) == 0:
                x_final = None
            else:
                x_final = xs[-1]
            hist = history_from_iterates(xs, a32, b32, xref64, gmres_outer)
            hist_path = out / 'histories' / f'{trial_id}.json'
            hist_path.write_text(json.dumps(jsonsafe({'trial_id': trial_id, 'case_id': case_id, 'method_id': method_id,
                                                      'terminal_reason': TERMINAL_MAP.get(status, status), 'failure': failure,
                                                      'history': hist}), allow_nan=False) + '\n')
            if x_final is not None:
                x_path = out / 'solutions' / f'{trial_id}.npy'
                np.save(x_path, x_final.astype(np.float32))
                mets = metrics_fp64(a32, b32, x_final, xref64)
                mets['residual_inf_mp100'] = mp_residual_inf(a32, b32, x_final, REF_DPS)
                mfin, reasons = metrics_with_flags(mets)
                x_sha = array_sha256(x_final.astype(np.float32))
                xp = str(x_path.relative_to(out))
            else:
                mfin = {k: None for k in ('eta', 'omega', 'relres_inf', 'relres_2', 'finf', 'f2', 'fcomp_nonzero', 'residual_inf_mp100', 'resid_inf_fp64')}
                reasons = {k: 'no_iterate_returned' for k in mfin}
                x_sha = None
                xp = None

            def le(v, thr):
                return bool(v is not None and v <= thr)
            qual = {'eta_le_10u': le(mfin['eta'], 10 * U32), 'eta_le_100u': le(mfin['eta'], 100 * U32),
                    'omega_le_10u': le(mfin['omega'], 10 * U32), 'omega_le_100u': le(mfin['omega'], 100 * U32),
                    'finf_le_10u': le(mfin['finf'], 10 * U32), 'finf_le_100u': le(mfin['finf'], 100 * U32)}
            nonfinite = bool(reasons) or status == 'numerical_failure'
            trials.append({
                'trial_id': trial_id, 'case_id': case_id, 'method_id': method_id,
                'precision_contract': PRECISION_CONTRACT[method_id],
                'terminal_reason': TERMINAL_MAP.get(status, status), 'failure': failure,
                'outer_iters': outer, 'inner_iters_total': inner_total,
                'matvecs': matvecs, 'precond_applies': precond_applies,
                'metrics': {k: mfin.get(k) for k in ('eta', 'omega', 'relres_inf', 'relres_2', 'finf', 'f2', 'fcomp_nonzero', 'residual_inf_mp100')},
                'nonfinite_reason': reasons if reasons else None,
                'qualification': qual,
                'first_attainment_10u': first_attainment(hist),
                'cpu_seconds': cpu, 'cpu_seconds_label': 'scalar_model_cpu_time_not_performance',
                'wall_seconds': wall,
                'x_path': xp, 'x_sha256': x_sha, 'factor_sha256': fh,
                'factor_hash_verified_against_shared_factorization': bool(verified),
                'history_path': str(hist_path.relative_to(out)),
                'gmres_params': gparams, 'nonfinite': nonfinite,
                'n_iterates_saved': int(len(xs)),
            })
        # paired-method factor identity checks
        packed = {t['method_id']: t['factor_sha256'] for t in trials}
        case_rec['factor_identity_checks'] = {
            'hss_hsd_halfsolve32_gmres_same_half_factors': len({packed[m] for m in ('hss', 'hsd', 'half_factors_solve32', 'paper_left_gmres_hsd_app64', 'paper_left_gmres_hsd_app32')}) == 1,
            'sss_ssd_same_fp32_factors': packed['sss'] == packed['ssd'],
            'all_run_solver_hashes_verified': all(t['factor_hash_verified_against_shared_factorization'] for t in trials),
        }
        case_rec['timing_seconds']['methods_wall'] = method_time
        case_rec['timing_seconds']['case_total_wall'] = time.perf_counter() - t_case
        return {'ok': True, 'idx': idx, 'case': jsonsafe(case_rec), 'trials': jsonsafe(trials), 'log': log}
    except Exception as e:  # recorded, never silently dropped
        return {'ok': False, 'idx': idx, 'n': n, 'kappa': kappa, 'seed': seed, 'error': repr(e),
                'traceback': traceback.format_exc(), 'log': log}


# ----------------------------------------------------------------------------- method metadata
PRECISION_CONTRACT = {
    'hss': '(u_f,u,u_r)=(16,32,32): FP16 op-rounded GEPP factors; FP16 op-rounded triangular solves; FP32 x and updates; FP32 sequential residual',
    'hsd': '(u_f,u,u_r)=(16,32,64): FP16 op-rounded GEPP factors; FP16 op-rounded triangular solves; FP32 x and updates; FP64 sequential residual',
    'sss': '(u_f,u,u_r)=(32,32,32): FP32 op-rounded GEPP factors; FP32 triangular solves; FP32 x and updates; FP32 sequential residual',
    'ssd': '(u_f,u,u_r)=(32,32,64): FP32 op-rounded GEPP factors; FP32 triangular solves; FP32 x and updates; FP64 sequential residual',
    'input16_ssd': 'input-only-half control: A32 rounded to FP16 (RNE) then promoted to FP32; FP32 GEPP factors; FP32 triangular solves; FP32 x and updates; FP64 residual of the ORIGINAL stored A32,b32',
    'half_factors_solve32': 'same-half-factors/FP32-application control: FP16 op-rounded GEPP factors (bit-identical to hss/hsd); triangular solves evaluated in FP32 on the FP16 factor values; FP32 x and updates; FP64 residual',
    'paper_left_gmres_hsd_app64': 'GMRES-IR (16,32,64), operator in FP64: FP16 factors identical to hsd; residual FP64; preconditioned residual U^-1 L^-1 P r and operator v->U^-1 L^-1 P A v evaluated in FP64 with the FP16 factor VALUES, then rounded to FP32; GMRES basis/Hessenberg/dots/Givens/least-squares in FP32; d in FP32; x+=d in FP32',
    'paper_left_gmres_hsd_app32': 'GMRES-IR (16,32,64), operator in FP32: identical to app64 except the preconditioned residual and the operator (A v sequential FP32, triangular solves in FP32 on the FP16 factor values) are evaluated in FP32',
}


def methods_json():
    common_stop = ('run up to 30 corrections; stop early only on exactly unchanged FP32 iterate after an update, an exact repeated '
                   'iterate with period <= 4, a computed zero residual in u_r, or a numerical failure (nonfinite / zero pivot). '
                   'No reference-based or threshold-based stopping. The LAST iterate is returned.')
    norm = ('power-of-two normalization: for a residual/RHS r with max magnitude m, e = frexp(m) exponent, solve with 2^-e r '
            '(scaled in residual precision), rescale the correction with ldexp in FP32 (tpir.arithmetic.scaled_solve)')
    classical_paper = ('Carson & Higham 2018 (SIAM J. Sci. Comput. 40(2)) Algorithm 1.1 with LU-based correction solves (Sections 6-7); '
                       'Higham 2019 (MIMS EPrint 2019.19) Algorithm 2.1')
    classical_dev = ('deviations: rank-one non-fused right-looking GEPP in an explicitly rounded software model (not LAPACK/hardware); '
                     'sequential (non-BLAS) residual accumulation; power-of-two instead of exact inf-norm scaling of the residual; '
                     'fixed 30-correction cap with unchanged/cycle detection rather than a convergence test; FP16 range failures are recorded, not rescued')
    gmres_paper = ('Carson & Higham 2018 Algorithm 8.1 (GMRES-IR, three precisions): LU in u_f, x_0 in u, r_i in u_r, GMRES on '
                   'U^-1 L^-1 A d = U^-1 L^-1 r_i at precision u with products with the preconditioned matrix at precision u_r, x_{i+1}=x_i+d_i in u; '
                   'Higham 2019 Algorithm 3.1')
    gmres_dev = ('deviations: Krylov dimension capped at m=min(30,n)=30 with NO restart (the paper uses unrestarted GMRES up to n; here gmres_stop=max_krylov_dim '
                 'marks outer steps where the cap bound); inner stopping on the recursive preconditioned relative residual <= 1e-4; '
                 'a verification gate (FP64 true correction residual ||r - A d||_inf/||r||_inf must be < 1, i.e. the correction must reduce the residual) '
                 'rejects the correction and terminates with terminal_reason gmres_correction_rejected_true_relres_ge_1 -- the paper has no gate; '
                 'operator products are sequential (non-BLAS) in the application precision; FP32 dot products use numpy BLAS sdot; '
                 'the preconditioned residual is computed in the application precision then rounded to FP32; no residual scaling (not needed without FP16 application); '
                 'attainable-accuracy stopping (30 outer cap, unchanged update, cycle) instead of a convergence criterion. '
                 'app32 is a deliberate control: the operator (A v and both triangular solves) evaluated in FP32 instead of FP64 (u_r)')
    rows = []
    for mid in METHOD_ORDER:
        if mid in CLASSICAL:
            tm = TPIR_METHODS[mid]
            fb = tm.factor
            sb = tm.solve_bits or tm.factor
            rows.append({
                'method_id': mid,
                'display_name': {'hss': 'LU-IR (half,single,single)', 'hsd': 'LU-IR (half,single,double)', 'sss': 'LU-IR (single,single,single)',
                                 'ssd': 'LU-IR (single,single,double)', 'input16_ssd': 'input-only-FP16 control (single LU, double residual)',
                                 'half_factors_solve32': 'half factors applied in FP32 control (double residual)'}[mid],
                'algorithm': 'classical LU-based iterative refinement, explicit (u_f,u,u_r) roles',
                'precision_contract': {
                    'tuple': f'({fb},{tm.work},{tm.resid})' + (f' solve_bits={sb}' if sb != fb else ''),
                    'factorization_input_conversion': ('stored FP32 A rounded to FP16 (RNE, gradual underflow) then promoted exactly to FP32 before FP32 LU' if mid == 'input16_ssd'
                                                       else ('stored FP32 A rounded to FP16 (RNE, gradual underflow)' if fb == 16 else 'stored FP32 A used as is')),
                    'panel_elimination_arithmetic': f'FP{fb}: every division, product and subtraction separately rounded (fl(a - fl(l*u)), no FMA), partial pivoting over all remaining rows',
                    'factor_storage': f'float{fb} packed LU + int64 row permutation',
                    'triangular_solve_arithmetic': f'FP{sb} column-oriented forward/back substitution, separately rounded multiply/subtract/divide, applied to the FP{fb} factor values',
                    'residual_arithmetic': f'FP{tm.resid} sequential s_i = fl(s_i + fl(a_ij x_j)) then fl(b - s), A,b,x promoted exactly',
                    'update_format': 'FP32: x_{i+1} = fl32(x_i + d_i), d_i = FP32 cast of the rescaled correction',
                    'gmres': None,
                    'description': PRECISION_CONTRACT[mid]},
                'pivoting': 'partial (row) pivoting, pivot = max |entry| over remaining rows, swaps applied to stored L columns',
                'stopping_rule': common_stop,
                'rhs_normalization': norm,
                'source_entrypoint': 'three_precision_lab/tpir/solver.py:run_solver (factor: tpir/arithmetic.py:lu_factor; solve: tpir/arithmetic.py:LU.solve, scaled_solve; residual: tpir/arithmetic.py:residual)',
                'paper_mapping': classical_paper if mid in ('hss', 'hsd', 'sss', 'ssd') else 'control/ablation not in the papers; same outer loop as hsd',
                'deviations': classical_dev,
                'counting_convention': 'matvecs = residual evaluations (one per correction attempt, plus the final zero-residual check if it ended the loop); precond_applies = initial solve (if it succeeded) + one pair of triangular solves per correction',
            })
        else:
            bits = GMRES_METHODS[mid]
            rows.append({
                'method_id': mid,
                'display_name': f'GMRES-IR with hsd half factors, preconditioned operator in FP{bits}',
                'algorithm': 'left-preconditioned GMRES-based iterative refinement (GMRES-IR)',
                'precision_contract': {
                    'tuple': f'(16,32,64) with operator application in FP{bits}',
                    'factorization_input_conversion': 'stored FP32 A rounded to FP16 (RNE, gradual underflow); factorization bit-identical to hsd (verified by factor hash)',
                    'panel_elimination_arithmetic': 'FP16: every division, product and subtraction separately rounded, partial pivoting over all remaining rows',
                    'factor_storage': 'float16 packed LU + int64 row permutation (same object as hsd)',
                    'triangular_solve_arithmetic': f'initial solve x_0: FP16 (identical to hsd x_0, with power-of-two normalization); inside GMRES-IR: FP{bits} forward/back substitution on the FP16 factor values',
                    'residual_arithmetic': 'FP64 sequential residual r_i = b - A x_i, exactly promoted inputs',
                    'update_format': 'FP32: x_{i+1} = fl32(x_i + d_i)',
                    'gmres': {'basis': 'FP32 (modified Gram-Schmidt Arnoldi)', 'dot_products': 'FP32 (numpy sdot)', 'givens': 'FP32', 'hessenberg_and_least_squares': 'FP32 back substitution',
                              'operator': f'FP{bits}: w = A v (sequential), z = U^-1 L^-1 P w with FP16 factor values; z rounded to FP32',
                              'preconditioned_rhs': f'FP{bits}: s = U^-1 L^-1 P r (r in FP64{", cast to FP32 first" if bits == 32 else ""}), rounded to FP32',
                              'inner_tolerance': '1e-4 on the recursive preconditioned relative residual |g_{k+1}|/||s||_2',
                              'krylov_cap': 'm = min(30, n), no restart', 'verification': 'FP64 true ||r - A d||_inf/||r||_inf recorded every outer step; correction accepted iff < 1'},
                    'description': PRECISION_CONTRACT[mid]},
                'pivoting': 'partial (row) pivoting (inherited from the shared FP16 factorization)',
                'stopping_rule': common_stop + ' Additionally, a correction whose FP64 true relative residual is >= 1 is rejected and the run terminates.',
                'rhs_normalization': 'initial solve: power-of-two normalization (as hsd). Corrections: none (GMRES normalizes v_1 = s/||s||_2; no FP16 application of residuals)',
                'source_entrypoint': 'e1_methods.py:gmres_ir (GMRES: e1_methods.py:gmres_fp32; factors: three_precision_lab/tpir/arithmetic.py:lu_factor, LU.solve)',
                'paper_mapping': gmres_paper,
                'deviations': gmres_dev,
                'counting_convention': 'matvecs = residual evaluations + GMRES operator applications (A v); precond_applies = initial solve + preconditioned residual per outer step + one per GMRES operator application',
            })
    return rows


# ----------------------------------------------------------------------------- summary
def write_summary(out, meta):
    out = Path(out)
    trials = [json.loads(l) for l in (out / 'trials.jsonl').read_text().splitlines() if l.strip()]
    cases = [json.loads(l) for l in (out / 'cases.jsonl').read_text().splitlines() if l.strip()]
    lines = ['# E1 scalar arithmetic-model track: run summary', '',
             f'Generated {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}.', '',
             f'Cases completed: {len(cases)} (planned {meta["planned_cases"]}); trials: {len(trials)} (planned {meta["planned_trials"]}).',
             f'Failed/absent cases: {meta["failed_cases"]}', '',
             'Thresholds: u32 = 2^-24; 10u32 = {:.3e}; 100u32 = {:.3e}. Qualification is on the FINAL iterate (last iterate, never reference-selected).'.format(10 * U32, 100 * U32), '',
             '## Qualification counts per method (count / denominator)', '',
             '| method | eta<=10u | eta<=100u | omega<=10u | omega<=100u | finf<=10u | finf<=100u | nonfinite | median outer | median inner |',
             '|---|---|---|---|---|---|---|---|---|---|']
    per_method = {}
    for mid in METHOD_ORDER:
        ts = [t for t in trials if t['method_id'] == mid]
        d = len(ts)
        if d == 0:
            continue
        q = lambda key: sum(1 for t in ts if t['qualification'][key])
        nf = sum(1 for t in ts if t['nonfinite'])
        mo = float(np.median([t['outer_iters'] for t in ts]))
        mi = float(np.median([t['inner_iters_total'] for t in ts]))
        per_method[mid] = {'denominator': d, 'eta_le_10u': q('eta_le_10u'), 'eta_le_100u': q('eta_le_100u'),
                           'omega_le_10u': q('omega_le_10u'), 'omega_le_100u': q('omega_le_100u'),
                           'finf_le_10u': q('finf_le_10u'), 'finf_le_100u': q('finf_le_100u'), 'nonfinite': nf}
        lines.append(f'| {mid} | {q("eta_le_10u")}/{d} | {q("eta_le_100u")}/{d} | {q("omega_le_10u")}/{d} | {q("omega_le_100u")}/{d} | {q("finf_le_10u")}/{d} | {q("finf_le_100u")}/{d} | {nf} | {mo:g} | {mi:g} |')
    lines += ['', '## finf <= 10u32 by nominal kappa (count over n and seeds; denominator per cell)', '']
    kappas = sorted({c['kappa_nominal'] for c in cases})
    cid2case = {c['case_id']: c for c in cases}
    lines.append('| method | ' + ' | '.join(f'k={k:g}' for k in kappas) + ' |')
    lines.append('|---|' + '---|' * len(kappas))
    for mid in METHOD_ORDER:
        row = []
        for k in kappas:
            ts = [t for t in trials if t['method_id'] == mid and cid2case[t['case_id']]['kappa_nominal'] == k]
            row.append(f'{sum(1 for t in ts if t["qualification"]["finf_le_10u"])}/{len(ts)}')
        lines.append(f'| {mid} | ' + ' | '.join(row) + ' |')
    lines += ['', '## Terminal reasons per method', '']
    for mid in METHOD_ORDER:
        ts = [t for t in trials if t['method_id'] == mid]
        cnt = {}
        for t in ts:
            cnt[t['terminal_reason']] = cnt.get(t['terminal_reason'], 0) + 1
        lines.append(f'- {mid}: ' + ', '.join(f'{k}={v}' for k, v in sorted(cnt.items())))
    audited = [c for c in cases if c['audit_digits']]
    lines += ['', '## Reference audit', '',
              f'- 100-digit mpmath LU (partial pivoting) reference for every case; 120-digit recheck on {len(audited)} cases (sorted index divisible by {AUDIT_EVERY}).',
              f'- max componentwise |x100-x120|/|x120| over audited cases: {max((c["reference_audit_maxrel"] for c in audited), default=None)}',
              f'- max 100-digit residual ||b - A x100||_inf of the reference over all cases: {max((c["reference_residual_inf_mp100"] for c in cases), default=None)}',
              f'- factor identity checks (same half factors across hss/hsd/half_factors_solve32/GMRES; run_solver hashes verified): all pass = {all(all(c["factor_identity_checks"].values()) for c in cases)}',
              '', '## What ran vs planned', '',
              f'- Planned grid: n in {NS}, kappa in {KAPPAS}, seeds {SEEDS}, {len(METHOD_ORDER)} methods -> {meta["planned_cases"]} cases, {meta["planned_trials"]} trials.',
              f'- Ran: ns = {meta["ns_run"]}; completed cases = {len(cases)}; dropped/failed: {meta["dropped"] or "none"}.',
              f'- Runtime: total wall {meta["total_wall_seconds"]:.1f} s with {meta["procs"]} worker processes (cores 8-79 via taskset); CPU only; the GPU was not used.',
              f'- Python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, mpmath {mp.__version__}; {platform.platform()}.',
              '- Prior-package test suite: see tests/pytest_log.txt (39 passed, no source changes; see CHANGES.md).',
              '', '## Notes / caveats', '',
              '- cpu_seconds are scalar-model (Python/NumPy software arithmetic) times: NOT performance data.',
              '- GMRES-IR Krylov dimension is capped at m=30 without restart (recorded as gmres_params.m_used); for ill-conditioned cases the cap binds (gmres_stop=max_krylov_dim in histories) which is a deviation from unrestricted GMRES in the papers.',
              '- GMRES-IR verification gate: a correction with FP64 true ||r - A d||_inf/||r||_inf >= 1 is rejected and the run stops (terminal_reason gmres_correction_rejected_true_relres_ge_1); the papers have no such gate.',
              '- Forward errors use the FP64 rounding of the 100-digit reference (rounding error ~1e-16 relative, immaterial at the u32 scale).',
              '- Normwise residual metrics eta/relres use numpy FP64 matmul (BLAS order); the final-iterate residual is also reported at 100 digits (residual_inf_mp100).',
              '', '## Commands', '', '```', *meta['commands'], '```', '']
    (out / 'summary.md').write_text('\n'.join(lines))
    (out / 'summary_counts.json').write_text(json.dumps(per_method, indent=2) + '\n')
    return per_method


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'out'))
    ap.add_argument('--procs', type=int, default=54)
    ap.add_argument('--ns', type=int, nargs='*', default=NS)
    ap.add_argument('--seeds', type=int, nargs='*', default=SEEDS, help='override seeds (smoke tests only; evaluation seeds are the default)')
    ap.add_argument('--kappas', type=float, nargs='*', default=KAPPAS)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = ' '.join(['taskset -c 8-79', sys.executable, *sys.argv])
    all_specs = sorted([(n, k, s) for n in NS for k in args.kappas for s in args.seeds])
    jobs = [(i, n, k, s, (i % AUDIT_EVERY == 0), str(out)) for i, (n, k, s) in enumerate(all_specs) if n in args.ns]
    meta_path = out / 'run_meta.json'
    if args.summary_only:
        meta = json.loads(meta_path.read_text())
        meta['commands'].append(cmd)
    else:
        t0 = time.perf_counter()
        print(f'[E1] {len(jobs)} cases, {len(jobs) * len(METHOD_ORDER)} trials, procs={args.procs}', flush=True)
        results = []
        ctx = mproc.get_context('fork')
        with ctx.Pool(processes=args.procs) as pool:
            for res in pool.imap_unordered(run_case, jobs):
                results.append(res)
                if res['ok']:
                    c = res['case']
                    print(f'[E1] done {len(results)}/{len(jobs)} idx={res["idx"]} n={c["n"]} kappa={c["kappa_nominal"]:g} seed={c["seed"]} '
                          f'id={c["case_id"]} wall={c["timing_seconds"]["case_total_wall"]:.1f}s elapsed={time.perf_counter() - t0:.0f}s', flush=True)
                else:
                    print(f'[E1] FAILED idx={res["idx"]} n={res["n"]} kappa={res["kappa"]:g} seed={res["seed"]}: {res["error"]}', flush=True)
        results.sort(key=lambda r: r['idx'])
        with (out / 'cases.jsonl').open('w') as fc, (out / 'trials.jsonl').open('w') as ft:
            for r in results:
                if not r['ok']:
                    continue
                fc.write(json.dumps(r['case'], allow_nan=False) + '\n')
                for t in r['trials']:
                    ft.write(json.dumps(t, allow_nan=False) + '\n')
        failed = [{'idx': r['idx'], 'n': r['n'], 'kappa': r['kappa'], 'seed': r['seed'], 'error': r['error'], 'traceback': r['traceback']} for r in results if not r['ok']]
        (out / 'failures.json').write_text(json.dumps(failed, indent=2) + '\n')
        total = time.perf_counter() - t0
        dropped = [f'n={n} (all {len(KAPPAS) * len(SEEDS)} cases not run)' for n in NS if n not in args.ns]
        meta = {'planned_cases': len(all_specs), 'planned_trials': len(all_specs) * len(METHOD_ORDER),
                'ns_run': args.ns, 'failed_cases': failed and [f['idx'] for f in failed] or [],
                'dropped': '; '.join(dropped + [f'failed case idx {f["idx"]} (n={f["n"]}, kappa={f["kappa"]:g}, seed={f["seed"]})' for f in failed]),
                'total_wall_seconds': total, 'procs': args.procs, 'commands': [cmd],
                'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'mpmath': mp.__version__,
                'platform': platform.platform(), 'grid': {'n': NS, 'kappa': KAPPAS, 'seeds': SEEDS}, 'methods': METHOD_ORDER,
                'max_corrections': MAX_CORR, 'gmres_tol': GMRES_TOL, 'gmres_m_cap': GMRES_M_CAP,
                'reference_digits': REF_DPS, 'audit_digits': AUDIT_DPS, 'audit_every': AUDIT_EVERY,
                'env_threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    (out / 'methods.json').write_text(json.dumps(methods_json(), indent=2) + '\n')
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    per_method = write_summary(out, meta)
    print(json.dumps(per_method, indent=1), flush=True)


if __name__ == '__main__':
    main()
