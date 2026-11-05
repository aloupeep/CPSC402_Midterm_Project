"""Explicitly rounded elementary operations. No FP16 BLAS, FMA, or hidden FP32 solve.

Binary16 operations are evaluated on exact binary16 operands in binary64 and
rounded back after EACH elementary operation. This implements round-to-nearest,
ties-to-even with gradual underflow; it is not a model of every GPU instruction.
FP32/64 ufuncs operate with explicitly matching input/output dtypes. Matrix-vector
products and triangular solves use a documented sequential non-fused ordering.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DTYPES = {16: np.dtype('float16'), 32: np.dtype('float32'), 64: np.dtype('float64')}
UNIT_ROUNDOFF = {16: 2.0**-11, 32: 2.0**-24, 64: 2.0**-53}

class NumericalFailure(RuntimeError):
    """An expected numerical outcome, not a program/runtime exception."""


def finite(a, stage):
    if not np.isfinite(a).all():
        raise NumericalFailure(f'nonfinite:{stage}')


@dataclass(frozen=True)
class Arithmetic:
    bits: int

    def __post_init__(self):
        if self.bits not in DTYPES:
            raise ValueError('Supported precisions are 16, 32, and 64.')

    @property
    def dtype(self):
        return DTYPES[self.bits]

    def cast(self, x):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            return np.asarray(x, dtype=self.dtype)

    def _op(self, fn, x, y):
        x, y = self.cast(x), self.cast(y)
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            if self.bits == 16:
                return fn(x.astype(np.float64), y.astype(np.float64)).astype(self.dtype)
            return fn(x, y, dtype=self.dtype)

    def add(self, x, y): return self._op(np.add, x, y)
    def sub(self, x, y): return self._op(np.subtract, x, y)
    def mul(self, x, y): return self._op(np.multiply, x, y)
    def div(self, x, y): return self._op(np.divide, x, y)

    def ldexp(self, x, exponent):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            # The exponent is an integer, not a potentially unrepresentable scale.
            if self.bits == 16:
                return np.ldexp(self.cast(x).astype(np.float64), exponent).astype(self.dtype)
            return np.ldexp(self.cast(x), exponent).astype(self.dtype)


def matvec_sequential(a, x, bits):
    """s_i = fl(s_i + fl(a_ij*x_j)), j=0..n-1, each operation rounded."""
    f = Arithmetic(bits)
    a, x = f.cast(a), f.cast(x)
    s = np.zeros(a.shape[0], dtype=f.dtype)
    for j in range(a.shape[1]):
        s = f.add(s, f.mul(a[:, j], x[j]))
    return s


def residual(a, b, x, bits, mode='sequential'):
    f = Arithmetic(bits)
    if mode == 'sequential':
        ax = matvec_sequential(a, x, bits)
    elif mode == 'blas':
        if bits not in (32, 64):
            raise ValueError('Native BLAS residual is available only for FP32/64.')
        ax = f.cast(a) @ f.cast(x)
    else:
        raise ValueError(f'Unknown residual mode {mode}')
    return f.sub(f.cast(b), ax)


@dataclass
class LU:
    lu: np.ndarray
    permutation: np.ndarray  # A[permutation, :] approx L U
    bits: int
    growth: float
    input_hash: str = ''

    def solve(self, rhs, arithmetic_bits=None):
        """Column-oriented triangular substitution with separately rounded mul/sub.

        arithmetic_bits defaults to the factorization precision. Overriding it is
        a named ablation, NEVER silently used for the standard half solver.
        """
        f = Arithmetic(self.bits if arithmetic_bits is None else arithmetic_bits)
        l = f.cast(self.lu)
        z = f.cast(rhs)[self.permutation].copy()
        n = l.shape[0]
        for j in range(n - 1):
            z[j+1:] = f.sub(z[j+1:], f.mul(l[j+1:, j], z[j]))
        finite(z, 'forward_substitution')
        for j in range(n-1, -1, -1):
            if l[j,j] == 0: raise NumericalFailure(f'zero_pivot_solve:{j}')
            z[j] = f.div(z[j], l[j,j])
            z[:j] = f.sub(z[:j], f.mul(l[:j, j], z[j]))
        finite(z, 'back_substitution')
        return z


def lu_factor(a, bits):
    """Right-looking rank-one GEPP; pivot selection uses all remaining rows.

    Every division, multiply and subtraction is rounded in `bits`. Swaps include
    already stored L columns. No library LU call is used here.
    """
    import hashlib
    f = Arithmetic(bits)
    w = f.cast(a).copy()
    if w.ndim != 2 or w.shape[0] != w.shape[1]: raise ValueError('A must be square')
    finite(w, 'factor_input')
    n = len(w)
    p = np.arange(n)
    initial_max = float(np.max(np.abs(w.astype(np.float64))))
    seen_max = initial_max
    digest = hashlib.sha256(w.tobytes()).hexdigest()
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(w[k:,k])))
        if w[pivot,k] == 0: raise NumericalFailure(f'zero_pivot_factor:{k}')
        if pivot != k:
            w[[k,pivot],:] = w[[pivot,k],:]
            p[[k,pivot]] = p[[pivot,k]]
        if k+1 < n:
            w[k+1:,k] = f.div(w[k+1:,k], w[k,k])
            product = f.mul(w[k+1:,k,None], w[k,None,k+1:])
            w[k+1:,k+1:] = f.sub(w[k+1:,k+1:], product)
            finite(w[k+1:,k:], f'factor_update:{k}')
            seen_max = max(seen_max, float(np.max(np.abs(w[k+1:,k+1:].astype(np.float64)))))
    return LU(w, p, bits, seen_max/initial_max if initial_max else float('inf'), digest)


def scaled_solve(factor, rhs, work_bits, rhs_bits, normalize=True, solve_bits=None):
    """Power-of-two normalization before low-precision conversion; undo in u.

    Normalization arithmetic occurs in the RHS/residual precision. Recovered d
    and x are rounded to the WORKING precision before any outer update.
    """
    rhs_arith, work = Arithmetic(rhs_bits), Arithmetic(work_bits)
    r = rhs_arith.cast(rhs)
    finite(r, 'rhs')
    peak = float(np.max(np.abs(r)))
    exponent = int(np.frexp(peak)[1]) if normalize and peak else 0
    rn = rhs_arith.ldexp(r, -exponent)
    z = factor.solve(rn, arithmetic_bits=solve_bits)
    d = work.ldexp(work.cast(z), exponent)
    finite(d, 'rescaled_solution')
    return d, exponent
