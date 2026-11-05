"""Reproducible runs, atomic records, saved iterates, provenance, and HP audits."""
from __future__ import annotations
import json,hashlib,platform,sys,os,time,traceback,zipfile
from pathlib import Path
import numpy as np
import scipy
from . import __version__
from .solver import METHODS,run_solver
from .problems import make_problem,reference_solution,errors


def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return str(x)
    return x


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def source_hash():
    root=Path(__file__).resolve().parent
    h=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def expand_cases(cfg):
    if 'cases' in cfg:return cfg['cases']
    import itertools
    g=cfg['grid'];out=[]
    keys=list(g)
    for vals in itertools.product(*(g[k] for k in keys)):
        d=dict(zip(keys,vals));d.setdefault('family','spectral');d.setdefault('rhs','manufactured');out.append(d)
    return out


def run(config,out,resume=False):
    config=Path(config);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(config.read_text())
    if len(cfg.get('methods',[]))!=len(set(cfg.get('methods',[]))):raise ValueError('Duplicate methods')
    for m in cfg['methods']:
        if m not in METHODS:raise ValueError(f'Unknown method {m}')
    cfg_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    sh=source_hash()
    meta_path=out/'manifest.json'
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        if not resume:raise ValueError('Output already exists; use --resume or a new output directory')
        if old['config_sha256']!=cfg_hash or old['source_sha256']!=sh:
            raise ValueError('Config/source changed. Use a NEW output directory, not --resume.')
        current={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,
                 'longdouble_precision_bits':int(np.finfo(np.longdouble).nmant+1)}
        if any(old.get(k)!=v for k,v in current.items()):
            raise ValueError('Environment changed. Use a NEW output directory; do not mix machines/libraries in a resumed run.')
    else:
        write_json(meta_path,{'version':__version__,'python':sys.version,'platform':platform.platform(),
            'numpy':np.__version__,'scipy':scipy.__version__,'config_sha256':cfg_hash,'source_sha256':sh,
            'longdouble_precision_bits':int(np.finfo(np.longdouble).nmant+1),
            'execution':'CPU explicitly rounded numerical model; NOT GPU performance data',
            'stopping_policy':'cap / unchanged update / exact repeated iterate / computed zero residual / numerical failure',
            'evaluation_thresholds':{'forward_accuracy':[10,100],'backward_accuracy':10,'units':'working unit roundoff'},
            'reference':'extended-residual FP64 reference; high-precision audit supplied separately',
            'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}})
        write_json(out/'config.json',cfg)
        # Actual source snapshot makes archived runs independently inspectable.
        for p in Path(__file__).parent.rglob('*.py'):
            q=out/'source'/p.relative_to(Path(__file__).parent);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    cases=expand_cases(cfg);nr=0;runtime_errors=0
    for i,spec in enumerate(cases):
        a,b,xman,case=make_problem(spec);cid=case['case_id']
        existing=out/'cases'/f'{cid}.json'
        if existing.exists():
            prior=json.loads(existing.read_text())
            if any(prior.get(k)!=case[k] for k in ('A_sha256','b_sha256')):
                raise ValueError('Regenerated inputs differ from saved inputs. Use a new output directory.')
            if resume and all((out/'records'/f'{cid}__{m}.json').exists() for m in cfg['methods']):
                print(f'[{i+1}/{len(cases)}] complete case {cid}: unchanged',flush=True)
                continue
        xref,refmeta=reference_solution(a,b)
        case['reference']=refmeta
        case['cond_componentwise_x_estimate']=float(np.linalg.norm(np.abs(np.linalg.inv(a.astype(np.float64)))@np.abs(a.astype(np.float64))@np.abs(xref.astype(np.float64)),np.inf)/np.linalg.norm(xref,np.inf))
        write_json(out/'cases'/f'{cid}.json',case)
        np.savez_compressed(out/'cases'/f'{cid}.npz',a=a,b=b,x_manufactured=xman,
            xref64=xref.astype(np.float64),xref_text=np.array([np.format_float_scientific(v,precision=30) for v in xref]))
        print(f'[{i+1}/{len(cases)}] n={spec["n"]} kappa={spec["kappa"]:g} seed={spec["seed"]} id={cid}',flush=True)
        for method in cfg['methods']:
            target=out/'records'/f'{cid}__{method}.json'
            if resume and target.exists():continue
            try:
                rec,xs=run_solver(a,b,method,cfg.get('max_iterations',30))
                rec.update(case_id=cid,case=case,source_sha256=sh)
                rec['history']=[{'iteration':j,**errors(a,b,x,xref,METHODS[method].work)} for j,x in enumerate(xs)]
                rec['final']=rec['history'][-1] if len(xs) else None
                rec['returned_candidate']=bool(len(xs) and rec['status']!='numerical_failure')
                rec['first_forward_10u_iteration']=next((v['iteration'] for v in rec['history'] if v['forward_le_10u']),None)
                rec['first_backward_10u_iteration']=next((v['iteration'] for v in rec['history'] if v['backward_le_10u']),None)
                rec['first_forward_100u_iteration']=next((v['iteration'] for v in rec['history'] if v['forward_le_100u']),None)
                array_path=out/'solutions'/f'{cid}__{method}.npz';array_path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(array_path,iterates=xs)
                rec['iterates_path']=str(array_path.relative_to(out))
            except Exception as e:
                runtime_errors+=1
                rec={'case_id':cid,'method':method,'status':'runtime_error','error':repr(e),
                     'traceback':traceback.format_exc(),'case':case,'returned_candidate':False}
            write_json(target,rec);nr+=1
    counts={}
    records=list((out/'records').glob('*.json'))
    for p in records:
        d=json.loads(p.read_text());counts[d['status']]=counts.get(d['status'],0)+1
    complete={'expected_records':len(cases)*len(cfg['methods']),'actual_records':len(records),
             'new_records':nr,'statuses':counts,'runtime_errors':counts.get('runtime_error',0)}
    write_json(out/'completion.json',complete)
    print(json.dumps(complete,indent=2),flush=True)
    if complete['runtime_errors']:raise RuntimeError('Runtime errors recorded; inspect records before interpreting results.')
    if len(records)!=complete['expected_records']:raise RuntimeError('Incomplete results')


def highprec(out,dps=80,max_cases=6,all_cases=False):
    """Independent mpmath solve of stored A,b; no re-running/refactoring the solver."""
    import mpmath as mp
    out=Path(out)
    candidates=[json.loads(p.read_text()) for p in sorted((out/'cases').glob('*.json'))]
    # Deterministic selection BEFORE observing numerical results. Prefer small
    # seed-0 cases, spread over kappa, to make a portable independent check cheap.
    candidates.sort(key=lambda x:(x['n'],x['spec']['seed'],x['kappa_nominal'],x['case_id']))
    if not all_cases:
        teaching=[c for c in candidates if c['spec']['seed']==0 and c['kappa_nominal']==1000]
        teaching.sort(key=lambda c:(abs(c['n']-64),c['case_id']))
        first=teaching[:1]
        candidates=(first+[c for c in candidates if c not in first])[:max_cases]
    dest=out/'high_precision';dest.mkdir(exist_ok=True)
    all_records=[]
    for ci,c in enumerate(candidates):
        cid=c['case_id'];raw=np.load(out/'cases'/f'{cid}.npz',allow_pickle=False)
        print(f'HP {ci+1}/{len(candidates)}: n={c["n"]}, kappa={c["kappa_nominal"]:g}, id={cid}',flush=True)
        with mp.workdps(dps):
            a=mp.matrix([[mp.mpf(float(v)) for v in row] for row in raw['a']])
            b=mp.matrix([mp.mpf(float(v)) for v in raw['b']])
            ref=mp.lu_solve(a,b)
            n=len(b);an=max(sum(abs(a[i,j]) for j in range(n)) for i in range(n));bn=max(abs(v) for v in b)
            refnorm=max(abs(v) for v in ref)
            rows=[]
            for p in sorted((out/'records').glob(f'{cid}__*.json')):
                record=json.loads(p.read_text());m=record['method']
                if not record.get('iterates_path'):continue
                xs=np.load(out/record['iterates_path'],allow_pickle=False)['iterates']
                if not len(xs):continue
                x=mp.matrix([mp.mpf(float(v)) for v in xs[-1]])
                r=b-a*x;xn=max(abs(v) for v in x)
                fe=max(abs(x[i]-ref[i]) for i in range(n))/(refnorm or 1)
                ne=max(abs(v) for v in r)/(an*xn+bn or 1)
                ce=mp.mpf(0)
                for i in range(n):
                    den=sum(abs(a[i,j])*abs(x[j]) for j in range(n))+abs(b[i])
                    val=abs(r[i])/den if den else (mp.inf if r[i] else mp.mpf(0))
                    ce=max(ce,val)
                row={'case_id':cid,'method':m,'dps':dps,'n':n,'kappa_nominal':c['kappa_nominal'],
                     'forward_error':float(fe),'normwise_backward_error':float(ne),'componentwise_backward_error':float(ce),
                     'forward_error_decimal':mp.nstr(fe,35),'normwise_backward_error_decimal':mp.nstr(ne,35),
                     'componentwise_backward_error_decimal':mp.nstr(ce,35),
                     'record_status':record['status']}
                for key in ('forward_error','normwise_backward_error','componentwise_backward_error'):
                    row[f'recorded_{key}']=record['final'][key]
                rows.append(row)
            write_json(dest/f'{cid}.json',{'case':c,'dps':dps,'reference_decimal':[mp.nstr(v,dps) for v in ref],
                'exact_input_conversion':'stored FP32 data -> exact Python float -> mpf', 'results':rows})
            all_records.extend(rows)
    write_json(dest/f'summary_{dps}dps.json',all_records)
    return all_records


def collect(paths,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    roots=[Path(p).resolve() for p in paths]
    for r in roots:
        if not r.exists():raise FileNotFoundError(r)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        used=set()
        for root in roots:
            if root.name in used:raise ValueError('Input directories need distinct names')
            used.add(root.name)
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.resolve()!=out.resolve() and '__pycache__' not in p.parts:
                    z.write(p,Path(root.name)/p.relative_to(root))
    print(f'Created {out} ({out.stat().st_size:,} bytes)',flush=True)
