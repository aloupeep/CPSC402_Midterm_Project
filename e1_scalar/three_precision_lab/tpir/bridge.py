"""Optional native torch / mixed-blocked bridge. NOT a pure-FP16 LU experiment.

Pure half arithmetic is in arithmetic.py. This module retains FP32 panels,
factors and triangular solves; only selected trailing GEMM operands are FP16.
The outer working precision is now FP32 and the residual precision is explicit.
All timings are diagnostic, not comparative hardware performance claims.
"""
from __future__ import annotations
import json,hashlib,time,platform
from pathlib import Path
import numpy as np
from .experiments import write_json,source_hash,expand_cases
from .problems import make_problem,reference_solution,errors

BRIDGE_METHODS={
    'native32_w32_r32':('fp32',32),
    'native32_w32_r64':('fp32',64),
    'blocked32_w32_r64':('blocked32',64),
    'mixed16inputs_w32_r64':('blocked16',64),
}


def solve_bridge(a,b,name,device='cuda',block_size=16,max_iterations=30):
    import torch
    from .vendor.factors import make_factor,NumericalFailure
    if name not in BRIDGE_METHODS:raise ValueError(name)
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate your known V100-compatible environment; no CPU fallback is performed.')
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        if hasattr(torch.backends.cuda.matmul,'allow_fp16_reduced_precision_reduction'):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    kind,rbits=BRIDGE_METHODS[name]
    aw=torch.tensor(a,device=device,dtype=torch.float32)
    bw=torch.tensor(b,device=device,dtype=torch.float32)
    rt=torch.float64 if rbits==64 else torch.float32
    ar,br=aw.to(rt),bw.to(rt)
    rec={'method':name,'precision':{'factor':'mixed FP16/FP32' if kind=='blocked16' else 32,'work':32,'resid':rbits},
         'work_dtype':'float32','factor_dtype':'float32','solve_dtype':'float32','residual_dtype':str(rt),
         'iterations':0,'factorizations':0,'status':'not_started','failure':None,
         'implementation':'FP32 panels/factors/solves; optional FP16-input/FP32-accumulation Schur updates',
         'execution':device,'stopping_policy':'same fixed-cap/unchanged/cycle semantics as CPU model',
         'timing_kind':'diagnostic only; host diagnostic transfers included',
         'tensor_core_dispatch':'not inferred from dtype; see prior trace or profile a new run'}
    hist=[];t0=time.perf_counter()
    try:
        f=make_factor(aw,kind,block_size,'power2',True,diagnostic=True)
        rec['factorizations']=1;rec['factor_metadata']=f.metadata
        try:x=f.solve(bw).to(torch.float32)
        except NumericalFailure as e:
            x=torch.zeros_like(bw);rec['initial_solve_failure']=str(e)
        hist.append(x.detach().cpu().numpy().copy());rec['status']='iteration_limit'
        for k in range(max_iterations):
            r=br-ar@x.to(rt)
            if not torch.isfinite(r).all().item():raise NumericalFailure('nonfinite_residual')
            if not torch.any(r).item():rec['status']='computed_residual_zero';break
            d=f.solve(r).to(torch.float32)
            new=x+d
            if new.dtype!=torch.float32:raise AssertionError('Working dtype drift')
            if not torch.isfinite(new).all().item():raise NumericalFailure('nonfinite_update')
            h=new.detach().cpu().numpy().copy();hist.append(h);rec['iterations']=k+1
            if torch.equal(x,new):x=new;rec['status']='unchanged_working_update';break
            if any(np.array_equal(h,old) for old in hist[max(0,len(hist)-6):-2]):x=new;rec['status']='working_precision_cycle';break
            x=new
    except NumericalFailure as e:
        rec['status']='numerical_failure';rec['failure']=str(e)
    if device=='cuda':torch.cuda.synchronize()
    rec['wall_seconds_including_python']=time.perf_counter()-t0
    return rec,np.stack(hist) if hist else np.empty((0,len(b)),dtype=np.float32)


def run_bridge(config,out,device='cuda',resume=False):
    import torch,traceback
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback. Core CPU campaign is independent.')
    cfg=json.loads(Path(config).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=source_hash();ch=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    meta={'source_sha256':source,'config_sha256':ch,'version':'1.0.0','execution':device,
          'torch':torch.__version__,'cuda_build':torch.version.cuda,'platform':platform.platform(),
          'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
          'arithmetic_caution':'These are NOT the classical pure-half factorization results.'}
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if not resume or any(old[k]!=meta[k] for k in ('source_sha256','config_sha256','execution','torch','cuda_build','platform','gpu')):
            raise ValueError('Existing bridge run incompatible or --resume missing; choose a new output directory')
    else:write_json(out/'manifest.json',meta);write_json(out/'config.json',cfg)
    cases=expand_cases(cfg)
    for i,spec in enumerate(cases):
        a,b,xman,c=make_problem(spec);cid=c['case_id']
        existing=out/'cases'/f'{cid}.json'
        if existing.exists():
            prior=json.loads(existing.read_text())
            if any(prior.get(k)!=c[k] for k in ('A_sha256','b_sha256')):
                raise ValueError('Regenerated bridge inputs differ. Use a new output directory.')
            if resume and all((out/'records'/f'{cid}__{m}.json').exists() for m in cfg['methods']):
                print(f'Bridge case {cid}: complete, unchanged',flush=True);continue
        ref,rm=reference_solution(a,b);c['reference']=rm
        write_json(out/'cases'/f'{cid}.json',c)
        np.savez_compressed(out/'cases'/f'{cid}.npz',a=a,b=b,x_manufactured=xman,xref64=ref.astype(np.float64))
        print(f'Bridge {i+1}/{len(cases)} {cid}',flush=True)
        for m in cfg['methods']:
            dest=out/'records'/f'{cid}__{m}.json'
            if resume and dest.exists():continue
            try:
                rec,xs=solve_bridge(a,b,m,device,cfg.get('block_size',16),cfg.get('max_iterations',30))
                rec.update(case_id=cid,case=c,source_sha256=source)
                rec['history']=[{'iteration':j,**errors(a,b,x,ref,32)} for j,x in enumerate(xs)]
                rec['final']=rec['history'][-1] if len(xs) else None
                rec['returned_candidate']=bool(len(xs) and rec['status']!='numerical_failure')
                q=out/'solutions'/f'{cid}__{m}.npz';q.parent.mkdir(exist_ok=True)
                np.savez_compressed(q,iterates=xs);rec['iterates_path']=str(q.relative_to(out))
            except Exception as e:
                rec={'case_id':cid,'case':c,'method':m,'status':'runtime_error','error':repr(e),'traceback':traceback.format_exc(),'returned_candidate':False}
            write_json(dest,rec)
    rows=[json.loads(p.read_text()) for p in (out/'records').glob('*.json')]
    counts={}
    for r in rows:counts[r['status']]=counts.get(r['status'],0)+1
    write_json(out/'completion.json',{'expected_records':len(cases)*len(cfg['methods']),'actual_records':len(rows),'statuses':counts,'runtime_errors':counts.get('runtime_error',0)})
    if counts.get('runtime_error'):raise RuntimeError('Bridge runtime errors recorded; core numerical results remain intact.')
