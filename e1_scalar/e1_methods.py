"""E1 (scalar arithmetic-model track) additions on top of the prior `tpir` package.

Contents
  * mp_lu_solve      : mpmath LU with partial pivoting at a declared number of decimal
                       digits, on the EXACT stored FP32 values (float32 -> Python float -> mpf
                       is exact).  Pure-Python lists of mpf (faster than mp.matrix indexing).
  * mp_residual_inf  : 100-digit infinity-norm residual ||b - A x||_inf of a float32 iterate.
  * gmres_fp32       : non-restarted GMRES in FP32 (MGS Arnoldi, FP32 dot products, FP32
                       Givens rotations, FP32 Hessenberg / least-squares back substitution).
  * gmres_ir         : left-preconditioned GMRES-IR (Carson & Higham 2018 Alg. 8.1; Higham
                       2019 Alg. 3.1) using the SAME half factors as `hsd`.  The preconditioned
                       operator  v -> U^{-1} L^{-1} P A v  and the preconditioned residual are
                       evaluated with the stored FP16 factor VALUES in a declared application
                       precision (64 = paper-style, 32 = incomplete-specification control).
  * metrics_fp64     : eta, omega, relres, forward errors from the exact stored FP32 A, b.

Nothing here uses a reference solution to choose an iterate or to stop.
"""
from __future__ import annotations
import time
import numpy as np
import mpmath as mp

from tpir.arithmetic import (Arithmetic, NumericalFailure, lu_factor, scaled_solve,
                             residual, matvec_sequential, finite, UNIT_ROUNDOFF)

F32 = np.float32
F64 = np.float64
U32 = 2.0 ** -24


# --------------------------------------------------------------------------- reference
def mp_lu_solve(a32, b32, dps):
    """Solve A x = b with mpmath at `dps` decimal digits; LU with partial pivoting.

    The float32 entries are converted exactly (float32 -> float -> mpf).  Row
    operations are plain Python loops over lists of mpf.  Returns a list of mpf
    (precision `dps`) -- caller decides how to store it.
    """
    with mp.workdps(dps):
        n = len(b32)
        a = [[mp.mpf(float(v)) for v in row] for row in np.asarray(a32, dtype=np.float32)]
        x = [mp.mpf(float(v)) for v in np.asarray(b32, dtype=np.float32)]
        for k in range(n):
            p = max(range(k, n), key=lambda i: abs(a[i][k]))
            if p != k:
                a[k], a[p] = a[p], a[k]
                x[k], x[p] = x[p], x[k]
            rowk = a[k]
            akk = rowk[k]
            if akk == 0:
                raise ZeroDivisionError(f'exactly singular pivot at step {k}')
            for i in range(k + 1, n):
                ai = a[i]
                l = ai[k] / akk
                if l:
                    ai[k] = l
                    for j in range(k + 1, n):
                        ai[j] -= l * rowk[j]
                    x[i] -= l * x[k]
        for i in range(n - 1, -1, -1):
            ai = a[i]
            s = x[i]
            for j in range(i + 1, n):
                s -= ai[j] * x[j]
            x[i] = s / ai[i]
        return x


def mp_residual_inf(a32, b32, x32, dps=100):
    """||b - A x||_inf at `dps` digits from the exact stored FP32 A, b and FP32 iterate x."""
    with mp.workdps(dps):
        a = np.asarray(a32, dtype=np.float32)
        x = [mp.mpf(float(v)) for v in np.asarray(x32)]
        worst = mp.mpf(0)
        for i in range(a.shape[0]):
            s = mp.mpf(float(b32[i]))
            row = a[i]
            for j in range(a.shape[1]):
                s -= mp.mpf(float(row[j])) * x[j]
            worst = max(worst, abs(s))
        return float(worst)


# --------------------------------------------------------------------------- metrics
def _f(x):
    x = float(x)
    return x


def metrics_fp64(a32, b32, x32, xref64):
    """Backward/forward error metrics in FP64 from the exact stored FP32 problem.

    r = b - A x uses numpy FP64 matmul (BLAS reduction order; FP64 rounding is
    ~1e-16 relative and immaterial at the u32 scale of interest).
    omega convention: 0/0 -> 0, positive/0 -> inf.
    Forward errors use the FP64 rounding of the 100-digit reference.
    """
    a = np.asarray(a32, dtype=F64)
    b = np.asarray(b32, dtype=F64)
    x = np.asarray(x32, dtype=F64)
    xr = np.asarray(xref64, dtype=F64)
    with np.errstate(all='ignore'):
        r = b - a @ x
        nr_inf = float(np.max(np.abs(r))) if r.size else 0.0
        nr_2 = float(np.linalg.norm(r))
        na = float(np.linalg.norm(a, np.inf))
        nx = float(np.max(np.abs(x)))
        nb_inf = float(np.max(np.abs(b)))
        nb_2 = float(np.linalg.norm(b))
        den = na * nx + nb_inf
        eta = nr_inf / den if den > 0 else (0.0 if nr_inf == 0 else float('inf'))
        dc = np.abs(a) @ np.abs(x) + np.abs(b)
        comp = np.zeros_like(dc)
        np.divide(np.abs(r), dc, out=comp, where=dc != 0)
        comp[(dc == 0) & (r != 0)] = np.inf
        omega = float(np.max(comp))
        nxr_inf = float(np.max(np.abs(xr)))
        nxr_2 = float(np.linalg.norm(xr))
        finf = float(np.max(np.abs(x - xr)) / nxr_inf) if nxr_inf > 0 else float('inf')
        f2 = float(np.linalg.norm(x - xr) / nxr_2) if nxr_2 > 0 else float('inf')
        nz = xr != 0
        fcomp = float(np.max(np.abs(x[nz] - xr[nz]) / np.abs(xr[nz]))) if nz.any() else 0.0
        relres_inf = nr_inf / nb_inf if nb_inf > 0 else (0.0 if nr_inf == 0 else float('inf'))
        relres_2 = nr_2 / nb_2 if nb_2 > 0 else (0.0 if nr_2 == 0 else float('inf'))
    return {'eta': eta, 'omega': omega, 'relres_inf': relres_inf, 'relres_2': relres_2,
            'finf': finf, 'f2': f2, 'fcomp_nonzero': fcomp, 'resid_inf_fp64': nr_inf}


# --------------------------------------------------------------------------- GMRES (FP32)
def gmres_fp32(apply_op, s32, tol, m):
    """Non-restarted GMRES on  A_tilde d = s  with d_0 = 0, everything in FP32.

    apply_op : FP32 vector -> FP32 vector (the preconditioned operator, already
               rounded to FP32 by the caller's application-precision policy)
    s32      : FP32 right-hand side (preconditioned residual rounded to FP32)
    tol      : stop when |g_{k+1}| / ||s||_2 <= tol  (recursive, preconditioned residual)
    m        : maximum Krylov dimension (no restart)
    Returns d32, iterations, recursive relative residual, stop reason.
    """
    s32 = np.asarray(s32, dtype=F32)
    n = s32.size
    beta = F32(np.linalg.norm(s32))          # FP32 2-norm
    if not np.isfinite(beta):
        raise NumericalFailure('nonfinite:gmres_rhs')
    if beta == 0:
        return np.zeros(n, dtype=F32), 0, 0.0, 'zero_preconditioned_rhs'
    V = np.zeros((m + 1, n), dtype=F32)
    H = np.zeros((m + 1, m), dtype=F32)
    cs = np.zeros(m, dtype=F32)
    sn = np.zeros(m, dtype=F32)
    g = np.zeros(m + 1, dtype=F32)
    g[0] = beta
    V[0] = s32 / beta
    k_done = 0
    relres = 1.0
    reason = 'max_krylov_dim'
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        for k in range(m):
            w = apply_op(V[k])
            if w.dtype != F32:
                raise AssertionError('operator must return float32')
            for j in range(k + 1):                       # modified Gram-Schmidt, FP32 dots
                hjk = F32(np.dot(V[j], w))
                H[j, k] = hjk
                w = w - hjk * V[j]
            hk1 = F32(np.linalg.norm(w))
            if not np.isfinite(hk1) or not np.isfinite(H[:k + 1, k]).all():
                raise NumericalFailure('nonfinite:gmres_arnoldi')
            # apply stored Givens rotations to the new column
            for j in range(k):
                t = cs[j] * H[j, k] + sn[j] * H[j + 1, k]
                H[j + 1, k] = -sn[j] * H[j, k] + cs[j] * H[j + 1, k]
                H[j, k] = t
            # new rotation (FP32)
            denom = F32(np.hypot(H[k, k], hk1))
            if denom == 0:
                cs[k], sn[k] = F32(1), F32(0)
            else:
                cs[k], sn[k] = H[k, k] / denom, hk1 / denom
            H[k, k] = cs[k] * H[k, k] + sn[k] * hk1
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            k_done = k + 1
            relres = float(abs(g[k + 1]) / beta)
            if hk1 == 0:
                reason = 'happy_breakdown'
                break
            if relres <= tol:
                reason = 'tol_reached'
                break
            if k + 1 < m:
                V[k + 1] = w / hk1
        # back substitution R y = g (FP32)
        y = np.zeros(k_done, dtype=F32)
        for i in range(k_done - 1, -1, -1):
            s = g[i]
            for j in range(i + 1, k_done):
                s = s - H[i, j] * y[j]
            if H[i, i] == 0:
                raise NumericalFailure('zero_diagonal:gmres_lsq')
            y[i] = s / H[i, i]
        d = (y @ V[:k_done]).astype(F32)   # FP32 combination of the FP32 basis
    if d.dtype != F32:
        raise AssertionError('gmres correction dtype drift')
    return d, k_done, relres, reason


# --------------------------------------------------------------------------- GMRES-IR
def gmres_ir(a_stored, b_stored, fac, app_bits, tol=1e-4, m_cap=30, max_outer=30,
             reject_ratio=1.0):
    """Left-preconditioned GMRES-IR with precisions (u_f,u,u_r) = (16,32,64).

    fac       : tpir LU object with FP16 factors (shared bit-for-bit with hsd)
    app_bits  : 64 -> operator  v -> U^{-1}L^{-1}P(A v)  and preconditioned residual
                      evaluated in FP64 using the FP16 factor values (paper style, u_r)
                32 -> the same evaluated in FP32 (control)
    GMRES basis / Hessenberg / dots / Givens in FP32 (u).  Residual r = b - A x in FP64
    (sequential, same model as hsd).  Update x += d in FP32.
    Gate: the FP64 true correction residual ||r - A d||_inf / ||r||_inf is computed before
    accepting; the correction is accepted iff it is finite and < reject_ratio.  (The paper
    has no gate; this is a recorded deviation.  reject_ratio=1 only rejects corrections
    that do not reduce the residual at all.)
    """
    work = Arithmetic(32)
    a = work.cast(a_stored)
    b = work.cast(b_stored)
    n = b.size
    m = min(m_cap, n)
    a64 = a.astype(F64)
    out = {'app_bits': app_bits, 'tol': tol, 'm': m, 'max_outer': max_outer,
           'reject_ratio': reject_ratio, 'status': 'not_started', 'failure': None,
           'iterations': 0, 'inner_iters_total': 0, 'matvecs': 0, 'precond_applies': 0,
           'initialization': 'low_precision_solve', 'gmres_outer': []}
    hist = []
    t0 = time.process_time()

    def precond_apply(v, bits):
        return fac.solve(v, arithmetic_bits=bits)

    if app_bits == 64:
        def apply_op(v32):
            out['matvecs'] += 1
            out['precond_applies'] += 1
            w = matvec_sequential(a64, v32.astype(F64), 64)
            z = precond_apply(w, 64)
            return z.astype(F32)
    elif app_bits == 32:
        def apply_op(v32):
            out['matvecs'] += 1
            out['precond_applies'] += 1
            w = matvec_sequential(a, v32, 32)
            z = precond_apply(w, 32)
            return z.astype(F32)
    else:
        raise ValueError('app_bits must be 32 or 64')

    try:
        try:
            x, exp0 = scaled_solve(fac, b, 32, 32, True, None)   # identical to hsd's x_0
            out['precond_applies'] += 1
        except NumericalFailure as e:
            x = np.zeros(n, dtype=F32)
            out['initialization'] = 'zero_after_failed_initial_solve'
            out['initial_solve_failure'] = str(e)
        hist.append(x.copy())
        out['status'] = 'iteration_limit'
        for i in range(max_outer):
            r = residual(a, b, x, 64, 'sequential')      # FP64 sequential residual
            out['matvecs'] += 1
            finite(r, 'residual')
            if not np.any(r):
                out['status'] = 'computed_residual_zero'
                break
            if app_bits == 64:
                s = precond_apply(r, 64).astype(F32)
            else:
                s = precond_apply(r.astype(F32), 32).astype(F32)
            out['precond_applies'] += 1
            d, iters, relres_pre, reason = gmres_fp32(apply_op, s, tol, m)
            out['inner_iters_total'] += iters
            finite(d, 'gmres_correction')
            # FP64 verification in original coordinates
            rd = r - a64 @ d.astype(F64)
            nr = float(np.max(np.abs(r)))
            true_ratio = float(np.max(np.abs(rd)) / nr) if nr > 0 else 0.0
            accepted = bool(np.isfinite(true_ratio) and true_ratio < reject_ratio)
            out['gmres_outer'].append({'outer': i + 1, 'inner_iters': iters,
                                       'gmres_relres_preconditioned': relres_pre,
                                       'gmres_stop': reason,
                                       'true_corr_relres_inf_fp64': true_ratio,
                                       'true_corr_le_tol': bool(true_ratio <= tol),
                                       'accepted': accepted})
            if not accepted:
                out['status'] = 'correction_rejected_true_relres_ge_1'
                break
            new = work.add(x, d)
            finite(new, 'working_update')
            if new.dtype != work.dtype:
                raise AssertionError('Working update dtype drift')
            hist.append(new.copy())
            out['iterations'] = i + 1
            if np.array_equal(new, x):
                x = new
                out['status'] = 'unchanged_working_update'
                break
            if any(np.array_equal(new, old) for old in hist[max(0, len(hist) - 6):-2]):
                x = new
                out['status'] = 'working_precision_cycle'
                break
            x = new
    except NumericalFailure as e:
        out['status'] = 'numerical_failure'
        out['failure'] = str(e)
    out['cpu_seconds'] = time.process_time() - t0
    xs = np.stack(hist) if hist else np.empty((0, n), dtype=F32)
    return out, xs
