"""Data-first report. Plotting is a separate, reproducible optional command."""
from __future__ import annotations
import csv,json,math,statistics
from pathlib import Path
from .experiments import write_json


def csvwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def report(path):
    root=Path(path);out=root/'report';out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(p.read_text()) for p in sorted((root/'records').glob('*.json'))]
    if not records:raise ValueError('No records found')
    rows=[];hist=[]
    for r in records:
        c=r['case'];s=c['spec'];f=r.get('final') or {};prec=r.get('precision') or {}
        row={'case_id':r['case_id'],'method':r['method'],'family':s.get('family','spectral'),
             'n':c['n'],'seed':s['seed'],'kappa_nominal':c['kappa_nominal'],'kappa2_stored':c['kappa2_stored'],
             'kappa_inf_stored':c['kappa_inf_stored'],'rhs':s.get('rhs','manufactured'),'rhs_scale_pow2':s.get('rhs_scale_pow2',0),
             'factor_precision':prec.get('factor'),'working_precision':prec.get('work'),'residual_precision':prec.get('resid'),
             'status':r['status'],'returned_candidate':r['returned_candidate'],'iterations':r.get('iterations'),
             'first_forward_10u_iteration':r.get('first_forward_10u_iteration'),
             'first_backward_10u_iteration':r.get('first_backward_10u_iteration'),
             'factor_growth':r.get('growth'),'factor_hash':r.get('factor_hash'),**f}
        row['qualified_forward_10u']=bool(r['returned_candidate'] and f.get('forward_le_10u',False))
        row['qualified_forward_100u']=bool(r['returned_candidate'] and f.get('forward_le_100u',False))
        row['qualified_backward_10u']=bool(r['returned_candidate'] and f.get('backward_le_10u',False))
        rows.append(row)
        for h in r.get('history',[]):hist.append({'case_id':r['case_id'],'method':r['method'],'n':c['n'],'seed':s['seed'],'kappa_nominal':c['kappa_nominal'],**h})
    csvwrite(out/'endpoints.csv',rows);csvwrite(out/'histories.csv',hist)
    summaries=[]
    for m in sorted(set(x['method'] for x in rows)):
        subset=[x for x in rows if x['method']==m]
        summaries.append({'method':m,'cases':len(subset),'forward_10u':sum(x['qualified_forward_10u'] for x in subset),
            'forward_100u':sum(x['qualified_forward_100u'] for x in subset),
            'backward_10u':sum(x['qualified_backward_10u'] for x in subset),
            'numerical_failures':sum(x['status']=='numerical_failure' for x in subset),
            'runtime_errors':sum(x['status']=='runtime_error' for x in subset)})
    csvwrite(out/'method_summary.csv',summaries)
    pairs=[]
    byid={(r['case_id'],r['method']):r for r in rows}
    for cid in sorted(set(r['case_id'] for r in rows)):
        for a,b in [('hss','hsd'),('sss','ssd'),('hsd','ssd'),('hsd','input16_ssd'),('hsd','half_factors_solve32')]:
            if (cid,a) not in byid or (cid,b) not in byid:continue
            ra,rb=byid[cid,a],byid[cid,b]
            ea=float(ra.get('forward_error',float('nan')));eb=float(rb.get('forward_error',float('nan')))
            ratios_ok=ra['returned_candidate'] and rb['returned_candidate'] and math.isfinite(ea) and math.isfinite(eb) and eb>0
            pairs.append({'case_id':cid,'method_numerator':a,'method_denominator':b,
                          'n':ra['n'],'seed':ra['seed'],'kappa_nominal':ra['kappa_nominal'],
                          'kappa_inf_stored':ra['kappa_inf_stored'],'forward_numerator':ea,'forward_denominator':eb,
                          'forward_error_ratio':ea/eb if ratios_ok else '',
                          'ratio_eligible_both_returned':ratios_ok,
                          'same_factors':ra['factor_hash']==rb['factor_hash'] if ra['factor_hash'] and rb['factor_hash'] else False,
                          'numerator_qualified':ra['qualified_forward_10u'],'denominator_qualified':rb['qualified_forward_10u']})
    csvwrite(out/'paired_comparisons.csv',pairs)
    hp=[]
    for p in sorted((root/'high_precision').glob('summary_*dps.json')):
        hp.extend(json.loads(p.read_text()))
    csvwrite(out/'high_precision_audit.csv',hp)
    # A predetermined teaching case, not the best case searched after the run.
    eligible=[r for r in rows if r['method']=='hsd' and r['seed']==0 and r['kappa_nominal']==1000]
    eligible.sort(key=lambda r:(abs(r['n']-64),r['case_id']))
    teaching=eligible[0]['case_id'] if eligible else None
    if teaching:
        csvwrite(out/'teaching_trace.csv',[h for h in hist if h['case_id']==teaching])
        write_json(out/'teaching_manifest.json',{'case_id':teaching,'selection_rule':'seed 0, kappa=1000, n closest to 64; independent of measured success','data_kind':'actual measured CPU arithmetic-model iterates','methods_available':[r['method'] for r in rows if r['case_id']==teaching]})
    lines=['# Three-precision numerical results','',
        'These are explicitly rounded CPU arithmetic experiments, not GPU performance measurements.',
        'All methods solve the identical stored FP32 problem; work=64 methods promote its exact values.',
        'The forward reference is FP64 LU corrected with extended residuals; see the independent mpmath audit.',
        '10u/100u are declared reporting thresholds, not universal theorem constants. u is each method\'s working unit roundoff.',
        'Stopping status is distinct from accuracy. An iteration cap or cycle is not automatically an inaccurate solution.',
        '', '| Method | Cases | Forward <= 10u | Forward <= 100u | Both backward <= 10u | Numerical failures |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['method']} | {s['cases']} | {s['forward_10u']} | {s['forward_100u']} | {s['backward_10u']} | {s['numerical_failures']} |")
    lines+=['','## Interpretation rules','',
        'Compare hss with hsd to isolate residual precision; their factor hashes should match.',
        'Compare sss with ssd as an independent residual-precision control.',
        'Compare hsd with ssd to change factorization AND correction-solve precision, not just factorization.',
        'Use half_factors_solve32 to isolate correction arithmetic with the same half factors.',
        'input16_ssd is input-only rounding followed by FP32 LU/solves. It is NOT half arithmetic.',
        'The hdd/sdd working precision differs, so the desired accuracy differs too.',
        'All failures remain in endpoints.csv and method counts. Ratios omit unavailable/nonfinite candidates explicitly.',
        '',f'Predetermined teaching case: `{teaching}`.',
        '', 'Run `python -m tpir plot <run-directory>` to generate plots from these exact tables.',
        'No curves or rates should be attributed to Tensor Cores or a V100; these core kernels are software arithmetic models.']
    (out/'READOUT.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote tables, teaching history and readout to {out}',flush=True)
