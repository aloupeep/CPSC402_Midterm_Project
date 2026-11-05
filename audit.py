#!/usr/bin/env python3
"""Independent audit pass (run after the GPU queue): for every case directory, regenerate the stored problem in a fresh
process (build/mpaudit), verify the A/B hashes, recompute FP64 and double-double original-system metrics for every unique
saved final solution, dump A for small cases (embedded inputs), and cross-check a deterministic selection on the host with
numpy (float64 and x86 long double) from the dumped matrices. Writes audit/audit_results.json and audit/host_crosscheck.json."""
from __future__ import annotations
import json, os, subprocess, sys, time, hashlib
from pathlib import Path
import numpy as np
ROOT = Path('/home/worker/lab/campaign'); RUNS = ROOT / 'runs'; AUD = ROOT / 'audit'; AUD.mkdir(exist_ok=True)
ENV = dict(os.environ, LD_LIBRARY_PATH='/home/worker/lab/campaign/build_magma/magma-2.9.0/lib:/home/linuxbrew/.linuxbrew/lib', OPENBLAS_NUM_THREADS='8', OMP_NUM_THREADS='8')
EMBED_MAX_N = 512

def main():
    suites = sys.argv[1].split(',') if len(sys.argv) > 1 else ['E2', 'E2a', 'E3', 'E4', 'E5', 'E6', 'E8', 'dev']
    results = json.load(open(AUD / 'audit_results.json')) if (AUD / 'audit_results.json').exists() else {}
    log = open(AUD / 'audit_log.txt', 'a')
    case_dirs = [d for s in sorted(RUNS.iterdir()) if s.name in suites for d in sorted(s.iterdir()) if d.is_dir()]
    for cdir in case_dirs:
        cid = cdir.name
        if cid in results and results[cid].get('regenerated_a_sha256'): continue
        cfiles = sorted(cdir.glob('case_*.json')); jobs = sorted(cdir.glob('done_*.json'))
        if not cfiles or not jobs: continue
        cj = json.load(open(cfiles[0])); c = json.load(open(jobs[0]))['job']['case']
        sols = {}
        for tf in cdir.glob('trials_*.jsonl'):
            for line in open(tf):
                d = json.loads(line)
                if d.get('x_path') and d.get('x_sha256') and (cdir / d['x_path']).exists(): sols[d['x_sha256']] = (str(cdir / d['x_path']), d['x_dtype'])
        lst = AUD / f'xlist_{cid}.txt'
        with open(lst, 'w') as f:
            for sha, (p, dt) in sols.items(): f.write(f'{p} {dt}\n')
        cmd = ['taskset', '-c', '0-7', str(ROOT / 'build' / 'mpaudit'), '--case-id', cid, '--x-list', str(lst)] + sum([[f"--{k.replace('_','-')}", str(v)] for k, v in c.items()], [])
        xr = list(cdir.glob('Xref_*')); xe = list(cdir.glob('Xexact_*'))
        if xr: cmd += ['--xref', str(xr[0])]
        if xe: cmd += ['--xexact', str(xe[0])]
        dump = None
        if cj['n'] <= EMBED_MAX_N:
            dump = ROOT / 'writeup_results' / 'matrices' / (cid + ('.f32' if cj['dtype'] == 'fp32' else '.f64')); dump.parent.mkdir(parents=True, exist_ok=True); cmd += ['--dump-a', str(dump)]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=1800)
            lines = [json.loads(l) for l in p.stdout.splitlines() if l.startswith('{')]
        except Exception as e:
            lines = [dict(fatal=str(e))]
        rec = dict(case_id=cid, n=cj['n'], dtype=cj['dtype'], recorded_a_sha256=cj['a_sha256'], recorded_b_sha256=cj['b_sha256'], elapsed_s=time.time() - t0, solutions={}, embedded_a_path=str(dump.relative_to(ROOT / 'writeup_results')) if dump else None)
        for l in lines:
            if 'regenerated_a_sha256' in l: rec.update(regenerated_a_sha256=l['regenerated_a_sha256'], regenerated_b_sha256=l['regenerated_b_sha256'], regenerated_tile_sha256=l['regenerated_tile_sha256'], regen_seconds=l['gen_seconds'], audit_pid=l.get('audit_process_pid'))
            elif 'x_sha256' in l: rec['solutions'][l['x_sha256']] = {k: l[k] for k in l if k not in ('x_path',)}
            elif 'fatal' in l: rec['error'] = l['fatal']
        rec['a_hash_match'] = rec.get('regenerated_a_sha256') == cj['a_sha256']; rec['b_hash_match'] = rec.get('regenerated_b_sha256') == cj['b_sha256']
        if dump and dump.exists(): rec['embedded_a_sha256'] = hashlib.sha256(open(dump, 'rb').read()).hexdigest()
        results[cid] = rec
        log.write(f"{cid} regen_match={rec['a_hash_match']} sols={len(rec['solutions'])} {rec.get('error','')} {rec['elapsed_s']:.1f}s\n"); log.flush()
        json.dump(results, open(AUD / 'audit_results.json', 'w'))
    # ---- host cross-check with numpy on embedded (n<=512) cases: deterministic selection = every case with an embedded A
    cross = {}
    for cid, rec in results.items():
        if not rec.get('embedded_a_path'): continue
        ap = ROOT / 'writeup_results' / rec['embedded_a_path']
        if not ap.exists(): continue
        cdir = next((d for s in RUNS.iterdir() if s.is_dir() for d in [s / cid] if d.is_dir()), None)
        if cdir is None: continue
        n = rec['n']; dt = np.float32 if rec['dtype'] == 'fp32' else np.float64
        A = np.fromfile(ap, dtype=dt).reshape((n, n), order='F').astype(np.float64)
        bfile = next(cdir.glob('B_*')); B = np.fromfile(bfile, dtype=dt).astype(np.float64); nrhs = B.size // n; B = B.reshape((n, nrhs), order='F')
        anorm = np.abs(A).sum(axis=1).max(); AL = A.astype(np.longdouble); BL = B.astype(np.longdouble)
        out = {}
        for sha, s in rec['solutions'].items():
            xp = None
            for tf in cdir.glob('trials_*.jsonl'):
                for line in open(tf):
                    d = json.loads(line)
                    if d.get('x_sha256') == sha and d.get('x_path'): xp = (cdir / d['x_path'], d['x_dtype']); break
                if xp: break
            if not xp: continue
            X = np.fromfile(xp[0], dtype=np.float32 if xp[1] == 'float32' else np.float64).astype(np.float64).reshape((n, nrhs), order='F')
            R = B - A @ X; RL = BL - AL @ X.astype(np.longdouble)
            eta = [float(np.abs(R[:, c]).max() / (anorm * np.abs(X[:, c]).max() + np.abs(B[:, c]).max())) if np.isfinite(X[:, c]).all() else None for c in range(nrhs)]
            den = np.abs(A) @ np.abs(X) + np.abs(B)
            with np.errstate(divide='ignore', invalid='ignore'):
                om = [float(np.nanmax(np.where(np.abs(R[:, c]) == 0, 0.0, np.abs(R[:, c]) / den[:, c]))) if np.isfinite(X[:, c]).all() else None for c in range(nrhs)]
            etaL = [float(np.abs(RL[:, c]).max() / (anorm * np.abs(X[:, c]).max() + np.abs(B[:, c]).max())) if np.isfinite(X[:, c]).all() else None for c in range(nrhs)]
            gpu = s.get('eta_fp64'); ratio = [None if (g is None or e is None or e == 0) else g / e for g, e in zip(gpu or [], eta)]
            out[sha] = dict(eta_numpy_fp64=eta, omega_numpy_fp64=om, eta_numpy_longdouble=etaL, eta_gpu_fp64=gpu, eta_gpu_dd=s.get('eta_dd'), gpu_over_numpy_eta_ratio=ratio, longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant))
        cross[cid] = out
    json.dump(cross, open(AUD / 'host_crosscheck.json', 'w'))
    print('audited cases', len(results), 'regen mismatches', sum(1 for r in results.values() if not r.get('a_hash_match')), 'host-crosschecked', len(cross))

if __name__ == '__main__': main()
