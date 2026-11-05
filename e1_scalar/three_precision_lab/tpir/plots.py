"""Reproduce plots from CSV, without a GPU. No invented/interpolated measurements."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def plot(path):
    root=Path(path);r=root/'report';figs=r/'figures';figs.mkdir(parents=True,exist_ok=True)
    rows=read_csv(r/'endpoints.csv')
    labels={'hss':'(16,32,32)','hsd':'(16,32,64)','sss':'(32,32,32)','ssd':'(32,32,64)',
            'hdd':'(16,64,64)','sdd':'(32,64,64)','direct32':'Direct FP32','direct16':'Direct FP16',
            'input16_ssd':'FP16 input only; FP32 LU','half_factors_solve32':'FP16 factors; FP32 solves'}
    def finish(name,note):
        plt.figtext(.5,.012,note,ha='center',fontsize=9)
        plt.tight_layout(rect=(0,.065,1,1))
        plt.savefig(figs/(name+'.png'),dpi=180,bbox_inches='tight')
        plt.savefig(figs/(name+'.svg'),bbox_inches='tight');plt.close()
    trace=r/'teaching_trace.csv'
    if trace.exists():
        h=read_csv(trace)
        plt.figure(figsize=(10,5.8))
        for m in ['hss','hsd','sss','ssd']:
            s=sorted([x for x in h if x['method']==m],key=lambda x:int(x['iteration']))
            if not s:continue
            plt.semilogy([int(x['iteration']) for x in s],[max(float(x['forward_error']),1e-20) for x in s],marker='o',markersize=3,label=labels[m])
        plt.axhline(2.**-24,linestyle='--',label='FP32 unit roundoff')
        sample=h[0];plt.title(f"Residual precision changes attainable accuracy | n={sample['n']}, nominal κ₂=1000, seed 0")
        plt.xlabel('Number of correction updates');plt.ylabel('Relative forward error to stored-system reference');plt.legend()
        finish('01_convergence','Measured software-arithmetic iterates; case chosen by a fixed rule. Reference is independently audited when available.')
        plt.figure(figsize=(10,5.8))
        for m in ['hsd','hdd','sdd']:
            s=sorted([x for x in h if x['method']==m],key=lambda x:int(x['iteration']))
            if not s:continue
            plt.semilogy([int(x['iteration']) for x in s],[max(float(x['forward_error']),1e-20) for x in s],marker='o',markersize=3,label=labels[m])
        plt.axhline(2.**-24,linestyle='--',label='FP32 unit roundoff');plt.axhline(2.**-53,linestyle=':',label='FP64 unit roundoff')
        plt.xlabel('Number of correction updates');plt.ylabel('Relative forward error');plt.title('Working precision changes the accuracy target');plt.legend()
        finish('02_working_precision','(16,32,64) targets single accuracy, not double. FP64 residual alone does not imply FP64 final accuracy.')
    plt.figure(figsize=(10,5.8))
    anycurve=False
    for m in ['hss','hsd','sss','ssd']:
        subset=[x for x in rows if x['method']==m]
        if not subset:continue
        ks=sorted(set(float(x['kappa_nominal']) for x in subset))
        rates=[]
        for k in ks:
            s=[x for x in subset if float(x['kappa_nominal'])==k]
            rates.append(100*sum(x['qualified_forward_10u']=='True' for x in s)/len(s))
        plt.semilogx(ks,rates,marker='o',label=labels[m]);anycurve=True
    if anycurve:
        plt.ylim(-5,105);plt.xlabel('Nominal κ₂ before FP32 storage');plt.ylabel('Cases with final forward error ≤ 10u (%)')
        plt.title('Precision-role comparison across conditioning');plt.legend()
        finish('03_accuracy_coverage','All configured cases remain in the denominator. 10u is a reporting threshold, not a theorem guarantee; see actual κ∞ in CSV.')
    else:plt.close()
    plt.figure(figsize=(10,5.8))
    shown=0
    for m,marker in [('hss','o'),('hsd','x')]:
        s=[x for x in rows if x['method']==m and x.get('forward_error') and np.isfinite(float(x['forward_error'])) and x['returned_candidate']=='True']
        if s:
            plt.loglog([float(x['kappa_inf_stored']) for x in s],[max(float(x['forward_error']),1e-20) for x in s],linestyle='none',marker=marker,label=labels[m]);shown+=1
    if shown:
        plt.axhline(2.**-24,linestyle='--',label='FP32 unit roundoff');plt.xlabel('Measured κ∞ of stored FP32 matrix');plt.ylabel('Final relative forward error')
        plt.title('Same low-precision factors, different residual precision');plt.legend()
        finish('04_actual_conditioning','Finite returned candidates shown; failures/unavailable solutions are retained in the coverage plot and endpoints table.')
    else:plt.close()
    print(f'Saved figures to {figs}')
