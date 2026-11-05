#!/usr/bin/env python3
"""E0 regression: replay the prior teaching case (cb384647987dbe4e, n=64, nominal kappa2=1000) saved final iterates against
independently evaluated stored-system residuals (numpy FP64 and mpmath 60 digits) and the stored 100-digit reference."""
import numpy as np, json, glob, mpmath as mp
from pathlib import Path
TC = Path('/home/worker/lab/handoff/mixed_precision_writeup_handoff/prior_evidence/teaching_case')
OUT = Path('/home/worker/lab/campaign/e0_replay'); OUT.mkdir(exist_ok=True)
u32 = 2.0 ** -24
out = {}
for sub in ('core', 'cuda_bridge'):
    z = np.load(TC / sub / 'cases' / 'cb384647987dbe4e.npz'); A = z['a'].astype(np.float64); b = z['b'].astype(np.float64); n = A.shape[0]
    hp = json.load(open(TC / 'core' / 'high_precision' / 'cb384647987dbe4e.json')) if sub == 'core' else None
    xref = np.array([float(s) for s in z['xref_text']]) if 'xref_text' in z.files else z['xref64']
    mp.mp.dps = 60; Am = mp.matrix(A.tolist()); bm = mp.matrix(b.tolist())
    anorm = np.abs(A).sum(axis=1).max()
    for rec in sorted(glob.glob(str(TC / sub / 'records' / '*.json'))):
        r = json.load(open(rec)); name = Path(rec).stem
        sol = np.load(TC / sub / 'solutions' / (name + '.npz'))
        X = sol['iterates'][-1].astype(np.float64) if 'iterates' in sol.files else sol[sol.files[0]].astype(np.float64).ravel()
        res = b - A @ X; eta = np.abs(res).max() / (anorm * np.abs(X).max() + np.abs(b).max()); den = np.abs(A) @ np.abs(X) + np.abs(b)
        with np.errstate(divide='ignore', invalid='ignore'): om = float(np.nanmax(np.where(np.abs(res) == 0, 0, np.abs(res) / den)))
        rm = bm - Am * mp.matrix(X.tolist()); eta_mp = float(max(abs(v) for v in rm) / (anorm * np.abs(X).max() + np.abs(b).max()))
        finf = float(np.abs(X - xref).max() / np.abs(xref).max())
        fin = r.get('final', {}); rec_vals = {k: v for k, v in fin.items() if isinstance(v, (int, float))} if isinstance(fin, dict) else {}
        out[f'{sub}/{name}'] = dict(status=r.get('status'), iterations=r.get('iterations'), replay=dict(eta_fp64=float(eta), eta_mp60=eta_mp, omega_fp64=om, forward_inf_vs_xref=finf, forward_le_10u32=bool(finf <= 10 * u32), eta_le_10u32=bool(eta <= 10 * u32)), recorded_final=rec_vals)
json.dump(out, open(OUT / 'teaching_case_replay.json', 'w'), indent=1)
lines = ['# E0 teaching-case replay (cb384647987dbe4e, n=64)', '', '| record | status | iters | eta (numpy FP64) | eta (mpmath 60d) | omega | forward vs xref | forward<=10u32 |', '|---|---|---|---|---|---|---|---|']
for k, v in out.items(): lines.append(f"| {k} | {v['status']} | {v['iterations']} | {v['replay']['eta_fp64']:.3e} | {v['replay']['eta_mp60']:.3e} | {v['replay']['omega_fp64']:.3e} | {v['replay']['forward_inf_vs_xref']:.4e} | {v['replay']['forward_le_10u32']} |")
lines += ['', 'Recorded prior metrics (final dicts) are stored alongside in teaching_case_replay.json for comparison; the prior evidence (CONTEXT.md) states hsd final forward error 3.0260721054e-8 and hss 1.6534729676e-5.']
(OUT / 'TEACHING_CASE_REPLAY.md').write_text('\n'.join(lines)); print('\n'.join(lines))
