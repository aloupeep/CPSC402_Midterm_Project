from __future__ import annotations
import argparse,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Explicit three-precision LU refinement experiments (accuracy, not emulation speed).')
    sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--config',required=True);r.add_argument('--out',required=True);r.add_argument('--resume',action='store_true')
    r=sub.add_parser('report');r.add_argument('path')
    r=sub.add_parser('plot');r.add_argument('path')
    r=sub.add_parser('audit');r.add_argument('path');r.add_argument('--dps',type=int,default=80);r.add_argument('--max-cases',type=int,default=6);r.add_argument('--all',action='store_true')
    r=sub.add_parser('collect');r.add_argument('paths',nargs='+');r.add_argument('--out',required=True)
    r=sub.add_parser('doctor');r.add_argument('--out',default=None)
    r=sub.add_parser('bridge');r.add_argument('--out',required=True);r.add_argument('--device',choices=['cpu','cuda'],default='cuda');r.add_argument('--config',default='configs/bridge.json');r.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.command=='run':
        from .experiments import run
        run(a.config,a.out,a.resume)
    elif a.command=='report':
        from .report import report
        report(a.path)
    elif a.command=='plot':
        from .plots import plot
        plot(a.path)
    elif a.command=='audit':
        from .experiments import highprec
        highprec(a.path,a.dps,a.max_cases,a.all)
    elif a.command=='collect':
        from .experiments import collect
        collect(a.paths,a.out)
    elif a.command=='bridge':
        from .bridge import run_bridge
        run_bridge(a.config,a.out,a.device,a.resume)
    elif a.command=='doctor':
        import numpy as np,scipy,platform
        from .arithmetic import Arithmetic,lu_factor,scaled_solve
        from .experiments import write_json,source_hash
        from .solver import run_solver
        d={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,
           'source_sha256':source_hash(),'longdouble_bits':int(np.finfo(np.longdouble).nmant+1)}
        h=Arithmetic(16)
        d['half_ties_even']=bool(h.add(1.,2**-11)==1.)
        d['half_gradual_underflow']=bool(h.cast(2**-24)!=0)
        a0=np.array([[4,1],[1,3]],dtype=np.float32);b0=np.array([1,2],dtype=np.float32)
        rec,x=run_solver(a0,b0,'hsd')
        d['hsd_smoke_status']=rec['status'];d['hsd_work_float32']=x.dtype==np.float32
        d['hsd_residual_float64']=rec['residual_dtype']=='float64'
        d['hsd_solve_float16']=rec['solve_dtype']=='float16'
        d['hsd_factor_float16']=rec['factor_dtype']=='float16'
        d['residual_inf']=float(np.linalg.norm(b0.astype(float)-a0.astype(float)@x[-1].astype(float),np.inf))
        try:
            import torch
            d['torch']=torch.__version__;d['cuda_build']=torch.version.cuda;d['cuda_available']=torch.cuda.is_available()
            if d['cuda_available']:d['gpu']=torch.cuda.get_device_name(0)
        except ImportError:d['torch']='not installed (optional only)'
        if a.out:write_json(a.out,d)
        print(json.dumps(d,indent=2))
        if not all(d[k] for k in ['half_ties_even','half_gradual_underflow','hsd_work_float32','hsd_residual_float64','hsd_solve_float16','hsd_factor_float16']):
            raise RuntimeError('Arithmetic self-check failed')

if __name__=='__main__':main()
