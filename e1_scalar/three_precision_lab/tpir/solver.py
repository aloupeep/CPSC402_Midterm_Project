"""Classical LU iterative refinement, explicit (u_f,u,u_r) precision roles."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np
from .arithmetic import Arithmetic, NumericalFailure, lu_factor, scaled_solve, residual, finite

@dataclass(frozen=True)
class Method:
    name: str
    factor: int
    work: int
    resid: int
    direct: bool = False
    input16_only: bool = False
    solve_bits: int | None = None
    normalize: bool = True
    residual_mode: str = 'sequential'

    def __post_init__(self):
        if not (self.factor <= self.work <= self.resid):
            raise ValueError('Require factor bits <= work bits <= residual bits.')

METHODS = {
    'hss': Method('hss',16,32,32),
    'hsd': Method('hsd',16,32,64),
    'sss': Method('sss',32,32,32),
    'ssd': Method('ssd',32,32,64),
    'hdd': Method('hdd',16,64,64),
    'sdd': Method('sdd',32,64,64),
    'ddd': Method('ddd',64,64,64),
    'direct16': Method('direct16',16,32,32,direct=True),
    'direct32': Method('direct32',32,32,32,direct=True),
    'direct64': Method('direct64',64,64,64,direct=True),
    'input16_ssd': Method('input16_ssd',32,32,64,input16_only=True),
    'half_factors_solve32': Method('half_factors_solve32',16,32,64,solve_bits=32),
    'hsd_no_norm': Method('hsd_no_norm',16,32,64,normalize=False),
    'hss_blas': Method('hss_blas',16,32,32,residual_mode='blas'),
    'hsd_blas': Method('hsd_blas',16,32,64,residual_mode='blas'),
    'sss_blas': Method('sss_blas',32,32,32,residual_mode='blas'),
    'ssd_blas': Method('ssd_blas',32,32,64,residual_mode='blas'),
}


def run_solver(a_stored, b_stored, method, max_iterations=30):
    """No reference solution and no diagnostic high-precision residual used to stop.

    Run until the declared iteration cap, exact unchanged update, a repeated
    iterate (period <= 4), zero residual in u_r, or numerical failure. Do not stop
    on the first small backward error: that can conceal forward-error improvement.
    Returns the LAST iterate, not an oracle-selected best iterate.
    """
    if max_iterations < 0: raise ValueError('max_iterations must be nonnegative')
    if isinstance(method, str): method=METHODS[method]
    m = method
    work = Arithmetic(m.work)
    a, b = work.cast(a_stored), work.cast(b_stored)
    if a.shape != (b.size,b.size) or b.ndim != 1: raise ValueError('Expected square A and vector b')
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Input must be finite')
    t0=time.perf_counter()
    out={'method':m.name,'precision':asdict(m),'iterations':0,'factorizations':0,
         'initialization':'low_precision_solve', 'normalization_exponents':[],
         'factor_dtype':None,'work_dtype':str(work.dtype),'residual_dtype':str(Arithmetic(m.resid).dtype),
         'solve_dtype':str(Arithmetic(m.solve_bits or m.factor).dtype),
         'timing_kind':'software_arithmetic_diagnostic_NOT_hardware_benchmark',
         'status':'not_started', 'failure':None}
    hist=[]
    try:
        af = Arithmetic(16).cast(a).astype(np.float32) if m.input16_only else a
        fac=lu_factor(af,m.factor)
        out.update(factorizations=1, factor_dtype=str(fac.lu.dtype), growth=fac.growth,
                   factor_hash=__import__('hashlib').sha256(fac.lu.tobytes()+fac.permutation.tobytes()).hexdigest())
        try:
            x, exp=scaled_solve(fac,b,m.work,m.work,m.normalize,m.solve_bits)
            out['normalization_exponents'].append(exp)
        except NumericalFailure as e:
            if m.direct: raise
            # Explicit zero start, not silent refactorization (paper section 8).
            x=np.zeros(b.size,dtype=work.dtype)
            out['initialization']='zero_after_failed_initial_solve'
            out['initial_solve_failure']=str(e)
        hist.append(x.copy())
        if m.direct:
            out['status']='direct'
        else:
            out['status']='iteration_limit'
            for k in range(max_iterations):
                r=residual(a,b,x,m.resid,m.residual_mode)
                finite(r,'residual')
                if not np.any(r):
                    out['status']='computed_residual_zero'; break
                d,exp=scaled_solve(fac,r,m.work,m.resid,m.normalize,m.solve_bits)
                out['normalization_exponents'].append(exp)
                new=work.add(x,d)
                finite(new,'working_update')
                if new.dtype != work.dtype: raise AssertionError('Working update dtype drift')
                hist.append(new.copy());out['iterations']=k+1
                if np.array_equal(new,x):
                    x=new;out['status']='unchanged_working_update';break
                if any(np.array_equal(new,old) for old in hist[max(0,len(hist)-6):-2]):
                    x=new;out['status']='working_precision_cycle';break
                x=new
    except NumericalFailure as e:
        out['status']='numerical_failure';out['failure']=str(e)
    out['wall_seconds_including_python']=time.perf_counter()-t0
    xs=np.stack(hist) if hist else np.empty((0,b.size),dtype=work.dtype)
    return out,xs
