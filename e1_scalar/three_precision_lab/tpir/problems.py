"""Stored FP32 problems shared exactly by all precision comparisons."""
from __future__ import annotations
import hashlib,json
import numpy as np
import scipy.linalg as sla


def make_problem(spec):
    n=int(spec['n']);kappa=float(spec['kappa']);seed=int(spec['seed'])
    if n<2 or kappa<1:raise ValueError('n >= 2 and kappa >= 1 required')
    rng=np.random.default_rng(seed)
    family=spec.get('family','spectral')
    s=np.geomspace(1,1/kappa,n)
    if spec.get('spectrum')=='one_small':s=np.ones(n);s[-1]=1/kappa
    if family=='diagonal': a=np.diag(s)
    elif family in ('spectral','spd'):
        u,_=np.linalg.qr(rng.standard_normal((n,n)))
        v=u if family=='spd' else np.linalg.qr(rng.standard_normal((n,n)))[0]
        a=(u*s)@v.T
    else:raise ValueError(f'Unknown family {family}')
    # Materialize the target matrix BEFORE forming the RHS.
    aw=a.astype(np.float32)
    x_man=rng.standard_normal(n)
    rhs=spec.get('rhs','manufactured')
    bw=(aw.astype(np.float64)@x_man if rhs=='manufactured' else rng.standard_normal(n)).astype(np.float32)
    scale=int(spec.get('rhs_scale_pow2',0))
    bw=np.ldexp(bw,scale).astype(np.float32)
    x_man=np.ldexp(x_man,scale) if rhs=='manufactured' else np.full(n,np.nan)
    a64=aw.astype(np.float64); b64=bw.astype(np.float64)
    cond2=float(np.linalg.cond(a64,2));condinf=float(np.linalg.cond(a64,np.inf))
    canonical=json.dumps(spec,sort_keys=True,separators=(',',':'))
    case_id=hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return aw,bw,x_man, {'case_id':case_id,'spec':spec,'n':n,'kappa_nominal':kappa,
         'kappa2_stored':cond2,'kappa_inf_stored':condinf,'data_dtype':'float32',
         'A_sha256':hashlib.sha256(aw.tobytes()).hexdigest(),
         'b_sha256':hashlib.sha256(bw.tobytes()).hexdigest(),
         'reference_problem':'exact values of saved FP32 A and b, NOT the generating x'}


def reference_solution(a,b):
    """FP64 LU plus extended-residual refinement; audited by mpmath separately.

    This is a diagnostic numerical reference, not mathematical ground truth.
    The longdouble significand size and disagreement with FP64 are recorded.
    """
    a64=np.asarray(a,dtype=np.float64);b64=np.asarray(b,dtype=np.float64)
    fac=sla.lu_factor(a64)
    ld=np.longdouble
    al=a64.astype(ld);bl=b64.astype(ld)
    x=sla.lu_solve(fac,b64).astype(ld)
    initial=x.copy()
    for _ in range(6):
        r=bl-al@x
        d=sla.lu_solve(fac,r.astype(np.float64)).astype(ld)
        new=x+d
        if np.array_equal(new,x): break
        x=new
    r=bl-al@x
    den=np.linalg.norm(al,np.inf)*np.linalg.norm(x,np.inf)+np.linalg.norm(bl,np.inf)
    meta={'kind':'FP64_LU_plus_longdouble_residual_refinement',
          'longdouble_precision_bits':int(np.finfo(ld).nmant+1),
          'backward_error':float(np.linalg.norm(r,np.inf)/den) if den else 0.,
          'initial_to_refined_relative_difference':float(np.linalg.norm(x-initial,np.inf)/np.linalg.norm(x,np.inf))}
    return x,meta


def errors(a,b,x,xref,work_bits):
    from .arithmetic import UNIT_ROUNDOFF
    ld=np.longdouble
    a=np.asarray(a,dtype=ld);b=np.asarray(b,dtype=ld);x=np.asarray(x,dtype=ld);xref=np.asarray(xref,dtype=ld)
    r=b-a@x
    nr=np.linalg.norm(r,np.inf)
    denom=np.linalg.norm(a,np.inf)*np.linalg.norm(x,np.inf)+np.linalg.norm(b,np.inf)
    dc=np.abs(a)@np.abs(x)+np.abs(b)
    c=np.zeros_like(dc);np.divide(np.abs(r),dc,out=c,where=dc!=0)
    c[(dc==0)&(r!=0)]=np.inf
    nx=np.linalg.norm(xref,np.inf)
    fe=float(np.linalg.norm(x-xref,np.inf)/(nx if nx else 1))
    ne=float(nr/denom) if denom else (0. if nr==0 else float('inf'))
    ce=float(np.max(c))
    u=UNIT_ROUNDOFF[work_bits]
    return {'forward_error':fe,'normwise_backward_error':ne,'componentwise_backward_error':ce,
            'relative_residual':float(nr/(np.linalg.norm(b,np.inf) or 1)),
            'forward_error_over_u':fe/u, 'normwise_backward_error_over_u':ne/u,
            'componentwise_backward_error_over_u':ce/u,
            # Predeclared descriptive thresholds; not theorem constants/certificates.
            'forward_le_10u':bool(fe<=10*u), 'forward_le_100u':bool(fe<=100*u),
            'backward_le_10u':bool(ne<=10*u and ce<=10*u)}
