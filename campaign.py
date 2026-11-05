#!/usr/bin/env python3
"""Campaign orchestrator: builds job plans per suite, runs the native driver sequentially (single GPU owner), collects records.
Original work for the CPSC 542G mixed-precision LU / iterative-refinement project (2026 campaign)."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time, fcntl, shutil
from pathlib import Path

ROOT = Path('/home/worker/lab/campaign')
DRIVER = ROOT / 'build' / 'mpsolve'
RUNS = ROOT / 'runs'
ENV = dict(os.environ, LD_LIBRARY_PATH='/home/worker/lab/campaign/build_magma/magma-2.9.0/lib:/home/linuxbrew/.linuxbrew/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
           OPENBLAS_NUM_THREADS='8', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
TASKSET = ['taskset', '-c', '0-7']

CASE_KEYS = ['n', 'nrhs', 'family', 'spectrum', 'kappa', 'seed', 'dtype', 'rhs', 'ell', 'nugget', 'dim', 'intmax', 'dyadic_exp', 'grcar_k', 'global_exp', 'rowcol_span']
CASE_DEFAULTS = dict(spectrum='geometric', kappa=100.0, rhs='random', ell=0.25, nugget=1e-3, dim=3, intmax=255, dyadic_exp=8, grcar_k=3, global_exp=0, rowcol_span=0)

def case_dict(**kw):
    c = dict(CASE_DEFAULTS); c.update(kw)
    return {k: c[k] for k in CASE_KEYS}

def case_id(c):
    canon = json.dumps({k: c[k] for k in CASE_KEYS}, sort_keys=True)
    h = hashlib.sha256(canon.encode()).hexdigest()[:10]
    kap = ('k%g' % c['kappa']).replace('+', '') if c['family'] == 'prescribed' else ''
    extra = ''
    if c['family'] == 'rbf': extra = 'l%g_nug%g' % (c['ell'], c['nugget'])
    if c['global_exp']: extra += '_g%+d' % c['global_exp']
    if c['rowcol_span']: extra += '_rc%d' % c['rowcol_span']
    return f"{c['family']}_{c['spectrum'] if c['family']=='prescribed' else c['rhs']}_{c['dtype']}_n{c['n']}_{kap}{extra}_s{c['seed']}_r{c['nrhs']}_{h}".replace('__', '_')

def job(suite, case, methods, target=None, reps=1, warmup=1, modes='resident', phase='evaluation', tag='', run_kind='timed_resident', **flags):
    c = case_dict(**case)
    return dict(suite=suite, case=c, case_id=case_id(c), methods=methods, target=target or c['dtype'], reps=reps, warmup=warmup, modes=modes, phase=phase, tag=tag, run_kind=run_kind, flags=flags)

def driver_cmd(j, out):
    c = j['case']
    cmd = TASKSET + [str(DRIVER), 'run', '--out', str(out), '--case-id', j['case_id'], '--n', str(c['n']), '--nrhs', str(c['nrhs']), '--family', c['family'], '--spectrum', c['spectrum'], '--kappa', repr(float(c['kappa'])),
           '--seed', str(c['seed']), '--dtype', c['dtype'], '--rhs', c['rhs'], '--ell', repr(float(c['ell'])), '--nugget', repr(float(c['nugget'])), '--dim', str(c['dim']), '--intmax', str(c['intmax']), '--dyadic-exp', str(c['dyadic_exp']),
           '--grcar-k', str(c['grcar_k']), '--global-exp', str(c['global_exp']), '--rowcol-span', str(c['rowcol_span']), '--target', j['target'], '--methods', ','.join(j['methods']), '--reps', str(j['reps']), '--warmup', str(j['warmup']),
           '--modes', j['modes'], '--phase', j['phase'], '--tag', j['tag'] or 'x']
    for k, v in j['flags'].items():
        if v is True: cmd += ['--' + k]
        elif v is False or v is None: continue
        else: cmd += ['--' + k, str(v)]
    return cmd

def run_jobs(jobs, suite_dir, log, timeout=3600):
    """Run sequentially; skip jobs whose done-marker exists. Holds an exclusive GPU lock."""
    suite_dir.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / 'gpu.lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX)
    for i, j in enumerate(jobs):
        jid = hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest()[:12]
        out = suite_dir / j['case_id']; out.mkdir(parents=True, exist_ok=True)
        marker = out / f'done_{jid}.json'
        if marker.exists(): continue
        cmd = driver_cmd(j, out)
        t0 = time.time(); log.write(f"[{time.strftime('%H:%M:%S')}] job {i+1}/{len(jobs)} {j['suite']} {j['case_id']} methods={len(j['methods'])}\n$ {' '.join(cmd)}\n"); log.flush()
        try:
            p = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=timeout)
            rc = p.returncode; err = p.stderr[-4000:]
        except subprocess.TimeoutExpired as e:
            rc = -9; err = f'TIMEOUT after {timeout}s'
        dt = time.time() - t0
        log.write(err + f"\n[rc={rc} elapsed={dt:.1f}s]\n"); log.flush()
        json.dump(dict(job=j, rc=rc, elapsed=dt, cmd=cmd, stderr_tail=err[-2000:], timestamp=time.time()), open(marker, 'w'))
    fcntl.flock(lock, fcntl.LOCK_UN)

# ---------------------------------------------------------------- method sets
T64_CORE = ['native_lu64', 'native_ir64_w64_r64', 'native_ir32_w64_r64', 'native_gmresir32_w64_r64', 'hetero16_ir_w64_r64', 'hetero16_gmresir_w64_r64', 'custom32_ir_w64_r64',
            'cusolver_irs_m64_l16_classical_fb_default', 'cusolver_irs_m64_l32_classical_fb_default', 'magma_lu64_native', 'magma_dx_sht_irstrs', 'magma_dx_ss_irstrs', 'magma_ht_ir_w64_r64', 'magma_ht_gmresir_w64_r64']
T64_NRHS1_ONLY = ['cusolver_irs_m64_l16_gmres_fb_default', 'magma_dx_sht_gmstrs']
T32_CORE = ['native_lu32', 'native_ir32_w32_r64', 'native_ir32_w32_r32', 'native_lu64_promoted32', 'native_gmresir32_w32_r64', 'hetero16_ir_w32_r64', 'hetero16_gmresir_w32_r64', 'custom32_ir_w32_r64',
            'cusolver_irs_m32_l16_classical_fb_default', 'cusolver_irs_m32_l32_classical_fb_default', 'magma_lu32_native']
T32_NRHS1_ONLY = ['cusolver_irs_m32_l16_gmres_fb_default']
POLICIES = ['policy_hetero16_stationary_only', 'policy_hetero16_gmres_only', 'policy_hetero16_escalate_immediate', 'policy_hetero16_budget30', 'policy_hetero16_budget120', 'policy_hetero16_budget480', 'policy_hetero16_cost_aware', 'policy_native32_escalate_immediate', 'policy_native32_budget120', 'native_lu64']

def methods_for(dtype, nrhs, large=False):
    if dtype == 'fp64':
        m = list(T64_CORE) + (T64_NRHS1_ONLY if nrhs == 1 else [])
        if large: m = [x for x in m if x not in ('custom32_ir_w64_r64', 'magma_dx_ss_irstrs', 'cusolver_irs_m64_l32_classical_fb_default', 'magma_ht_gmresir_w64_r64')]
    else:
        m = list(T32_CORE) + (T32_NRHS1_ONLY if nrhs == 1 else [])
        if large: m = [x for x in m if x not in ('custom32_ir_w32_r64', 'native_lu64_promoted32', 'cusolver_irs_m32_l32_classical_fb_default')]
    return m

# ---------------------------------------------------------------- suites
def plan_dev(frozen):
    J = []
    for seed in (9100, 9101):
        for n, reps in ((4096, 3), (8192, 2)):
            base = dict(n=n, nrhs=1, family='prescribed', spectrum='geometric', kappa=100.0, seed=seed, dtype='fp64')
            for nb in (128, 256, 512):
                J.append(job('dev', base, ['hetero16_ir_w64_r64', 'custom32_ir_w64_r64', 'hetero16_lu_direct', 'custom32_lu_direct'], reps=reps, warmup=1, phase='development', tag=f'nb{nb}', nb=nb))
            J.append(job('dev', base, ['native_lu64', 'native_lu32', 'native_ir32_w64_r64', 'native_ir64_w64_r64', 'cusolver_irs_m64_l16_classical_fb_default', 'cusolver_irs_m64_l32_classical_fb_default', 'cusolver_irs_m64_l16_gmres_fb_default', 'magma_lu64_native', 'magma_lu64_hybrid', 'magma_dx_sht_irstrs', 'magma_dx_ss_irstrs', 'magma_dx_sht_gmstrs', 'magma_ht_ir_w64_r64'], reps=reps, warmup=1, phase='development', tag='baselines', nb=256))
        # GMRES tuning on a hard case
        hard = dict(n=4096, nrhs=1, family='prescribed', spectrum='geometric', kappa=1e6, seed=seed, dtype='fp64')
        for m in (20, 40, 80):
            for tol in (1e-2, 1e-4, 1e-6):
                J.append(job('dev', hard, ['hetero16_gmresir_w64_r64', 'native_gmresir32_w64_r64'], reps=1, warmup=1, phase='development', tag=f'gm{m}_tol{tol:g}', **{'gmres-m': m, 'gmres-tol': tol, 'gmres-budget': 480, 'nb': 256}))
    return J

def frozen_flags(frozen):
    return {'nb': frozen['nb'], 'gmres-m': frozen['gmres_m'], 'gmres-tol': frozen['gmres_tol'], 'gmres-budget': frozen['gmres_budget'], 'max-outer': frozen['max_outer'], 'tau-mult': frozen['tau_mult']}

def plan_e2(frozen):
    J = []; F = frozen_flags(frozen)
    for seed in (9200, 9201, 9202):
        for n in (512, 2048):
            for dtype in ('fp64', 'fp32'):
                for sp in ('geometric', 'clustered', 'one_small'):
                    kappas = [1e2, 1e4, 1e6, 1e8] + ([1e10, 1e12] if (dtype == 'fp64' and sp == 'geometric') else [])
                    for k in kappas:
                        J.append(job('E2', dict(n=n, nrhs=1, family='prescribed', spectrum=sp, kappa=k, seed=seed, dtype=dtype), methods_for(dtype, 1), reps=1, warmup=1, phase='evaluation', tag='E2', **F))
                for ell, nug in ((0.1, 1e-2), (0.5, 1e-6)):
                    J.append(job('E2', dict(n=n, nrhs=1, family='rbf', ell=ell, nugget=nug, seed=seed, dtype=dtype), methods_for(dtype, 1), reps=1, warmup=1, phase='evaluation', tag='E2', **F))
                J.append(job('E2', dict(n=n, nrhs=2, family='dyadic', rhs='exact_sign', seed=seed, dtype=dtype), methods_for(dtype, 2), reps=1, warmup=1, phase='evaluation', tag='E2', **F))
                J.append(job('E2', dict(n=n, nrhs=1, family='grcar', seed=seed, dtype=dtype), methods_for(dtype, 1), reps=1, warmup=1, phase='evaluation', tag='E2', **F))
        for n in (64, 128):
            J.append(job('E2', dict(n=n, nrhs=1, family='wilkinson', seed=seed, dtype='fp32'), methods_for('fp32', 1), reps=1, warmup=1, phase='evaluation', tag='E2', **F))
        for n in (256, 512):
            J.append(job('E2', dict(n=n, nrhs=1, family='wilkinson', seed=seed, dtype='fp64'), methods_for('fp64', 1), reps=1, warmup=1, phase='evaluation', tag='E2', **F))
    # staged 8192 representatives (seed 9200 only): easy / boundary / hard
    for sp, k in (('geometric', 1e2), ('clustered', 1e4), ('geometric', 1e6), ('geometric', 1e8)):
        for dtype in ('fp64', 'fp32'):
            J.append(job('E2', dict(n=8192, nrhs=1, family='prescribed', spectrum=sp, kappa=k, seed=9200, dtype=dtype), methods_for(dtype, 1), reps=1, warmup=1, phase='evaluation', tag='E2_8192', **F))
    return J

E3_METHODS = ['native_lu64', 'native_ir32_w64_r64', 'native_ir32_w64_r64_global', 'native_ir32_w64_r64_twosided', 'hetero16_ir_w64_r64', 'hetero16_ir_w64_r64_noscale', 'hetero16_ir_w64_r64_twosided', 'hetero16_gmresir_w64_r64', 'cusolver_irs_m64_l16_classical_fb_default', 'magma_dx_sht_irstrs']
def plan_e3(frozen):
    J = []; F = frozen_flags(frozen)
    for seed in (9200, 9201, 9202):
        for n in (256, 2048):
            for g in (-60, -30, 0, 30, 60):
                J.append(job('E3', dict(n=n, nrhs=1, family='prescribed', spectrum='geometric', kappa=1e4, seed=seed, dtype='fp64', global_exp=g), E3_METHODS, reps=1, warmup=1, phase='evaluation', tag='E3_global', **F))
            for span in (8, 16, 24):
                J.append(job('E3', dict(n=n, nrhs=1, family='prescribed', spectrum='geometric', kappa=1e4, seed=seed, dtype='fp64', rowcol_span=span), E3_METHODS, reps=1, warmup=1, phase='evaluation', tag='E3_rowcol', **F))
    for g, span in ((0, 0), (30, 0), (0, 16)):
        J.append(job('E3', dict(n=8192, nrhs=1, family='prescribed', spectrum='geometric', kappa=1e4, seed=9200, dtype='fp64', global_exp=g, rowcol_span=span), E3_METHODS, reps=1, warmup=1, phase='evaluation', tag='E3_8192', **F))
    return J

def plan_e4(frozen, sizes64, sizes32):
    """Trimmed (time-limited) large-n plan, declared before any E4 timing was viewed. See configs/frozen_config.json repetitions/seeds."""
    J = []; F = frozen_flags(frozen)
    def reps_for(n): return 7 if n <= 8192 else (5 if n <= 16384 else 3)
    def seeds_for(n): return [9200, 9201, 9202] if n <= 8192 else ([9200, 9201] if n == 16384 else [9200])
    CORE64 = ['native_lu64', 'native_ir64_w64_r64', 'native_ir32_w64_r64', 'native_gmresir32_w64_r64', 'hetero16_ir_w64_r64', 'hetero16_gmresir_w64_r64', 'magma_ht_ir_w64_r64']
    CORE32 = ['native_lu32', 'native_ir32_w32_r64', 'native_ir32_w32_r32', 'native_gmresir32_w32_r64', 'hetero16_ir_w32_r64', 'hetero16_gmresir_w32_r64']
    for dtype, sizes in (('fp64', sizes64), ('fp32', sizes32)):
        for n in sizes:
            for seed in seeds_for(n):
                fams = (('geometric', 1e2), ('clustered', 1e4)) if (n <= 24576 or (n == 32768 and dtype == 'fp64')) else (('geometric', 1e2),)
                for sp, k in fams:
                    nrhs_list = (1, 8) if n <= 8192 else ((1, 8) if (n == 16384 and sp == 'geometric' and seed == 9200) else (1,))
                    for nrhs in nrhs_list:
                        c = dict(n=n, nrhs=nrhs, family='prescribed', spectrum=sp, kappa=k, seed=seed, dtype=dtype)
                        ms = methods_for(dtype, nrhs, n >= 24576)
                        if n >= 32768: groups = [[m for m in ms if m in (CORE64 if dtype == 'fp64' else CORE32)]] + ([[m for m in ms if m.startswith('cusolver_irs')], [m for m in ms if m.startswith('magma_') and m != 'magma_ht_ir_w64_r64']] if (n == 32768 and sp == 'geometric') else [])
                        else: groups = [ms]
                        host_reps = 2 if n <= 16384 else 1
                        for gi, g in enumerate(groups):
                            if not g: continue
                            J.append(job('E4', c, g, reps=reps_for(n), warmup=1, modes='resident', phase='evaluation', tag=f'E4res_g{gi}', **F))
                            J.append(job('E4', c, g, reps=host_reps, warmup=1, modes='host', phase='evaluation', tag=f'E4host_g{gi}', run_kind='timed_host', **F))
    return J

def plan_e5(frozen, sizes):
    J = []; F = frozen_flags(frozen)
    reuse_methods = ['native_lu64', 'native_ir64_w64_r64', 'native_ir32_w64_r64', 'native_gmresir32_w64_r64', 'hetero16_ir_w64_r64', 'hetero16_gmresir_w64_r64']
    for n in sizes:
        for nrhs in (1, 8, 32, 64):
            c = dict(n=n, nrhs=nrhs, family='prescribed', spectrum='geometric', kappa=1e2, seed=9200, dtype='fp64')
            J.append(job('E5', c, reuse_methods, reps=2, warmup=1, phase='evaluation', tag='E5reuse', run_kind='reuse', reuse=True, **F))
            J.append(job('E5', c, reuse_methods + ['cusolver_irs_m64_l16_classical_fb_default', 'magma_dx_sht_irstrs'], reps=2, warmup=1, phase='evaluation', tag='E5fresh', **F))
    return J

def plan_e6(frozen, sizes):
    J = []; F = frozen_flags(frozen)
    for n in sizes:
        for seed in (9200, 9201, 9202):
            for k in (1e6, 1e8):
                J.append(job('E6', dict(n=n, nrhs=1, family='prescribed', spectrum='geometric', kappa=k, seed=seed, dtype='fp64'), POLICIES, reps=1, warmup=1, phase='evaluation', tag='E6', **F))
            J.append(job('E6', dict(n=n, nrhs=1, family='rbf', ell=0.5, nugget=1e-8, seed=seed, dtype='fp64'), POLICIES, reps=1, warmup=1, phase='evaluation', tag='E6', **F))
    for n in (4096, 8192):
        for seed in (9300, 9301, 9302):
            for k in (1e6, 1e8):
                J.append(job('E6', dict(n=n, nrhs=1, family='prescribed', spectrum='geometric', kappa=k, seed=seed, dtype='fp64'), ['policy_hetero16_cost_aware', 'policy_hetero16_escalate_immediate', 'policy_hetero16_budget120', 'native_lu64'], reps=1, warmup=1, phase='confirmation', tag='E6conf', **F))
    return J

def plan_e2a(frozen):
    """Attainable-accuracy track: tau-mult 0.001 makes the backward tolerance unattainable so refinement runs until stagnation/max_outer (old unchanged-update style rule)."""
    J = []; F = dict(frozen_flags(frozen)); F['tau-mult'] = 0.001
    A32 = ['native_lu32', 'native_ir32_w32_r64', 'native_ir32_w32_r32', 'native_ir32_w64_r64', 'native_gmresir32_w32_r64', 'native_gmresir32_w64_r64', 'hetero16_ir_w32_r64', 'hetero16_ir_w64_r64', 'hetero16_gmresir_w64_r64', 'native_lu64_promoted32', 'custom32_ir_w64_r64']
    A64 = ['native_lu64', 'native_ir64_w64_r64', 'native_ir32_w64_r64', 'native_gmresir32_w64_r64', 'hetero16_ir_w64_r64', 'hetero16_gmresir_w64_r64']
    for seed in (9200, 9201, 9202):
        for n in (512, 2048):
            for k in (1e2, 1e4, 1e6):
                J.append(job('E2a', dict(n=n, nrhs=1, family='prescribed', spectrum='geometric', kappa=k, seed=seed, dtype='fp32'), A32, reps=1, warmup=1, phase='evaluation', tag='attain', run_kind='diagnostic', **F))
            for k in (1e4, 1e8, 1e12):
                J.append(job('E2a', dict(n=n, nrhs=1, family='prescribed', spectrum='geometric', kappa=k, seed=seed, dtype='fp64'), A64, reps=1, warmup=1, phase='evaluation', tag='attain', run_kind='diagnostic', **F))
    return J

def plan_e8(frozen):
    J = []; F = frozen_flags(frozen)
    for n in (8192, 32768):
        J.append(job('E8', dict(n=n, nrhs=1, family='prescribed', spectrum='geometric', kappa=1e2, seed=9200, dtype='fp64'), ['native_lu64', 'hetero16_ir_w64_r64', 'hetero16_gmresir_w64_r64', 'native_ir32_w64_r64'], reps=1, warmup=1, phase='evaluation', tag='E8prof', run_kind='profile', instrument=True, **F))
    J.append(job('E8', dict(n=8192, nrhs=1, family='prescribed', spectrum='geometric', kappa=1e6, seed=9200, dtype='fp64'), ['hetero16_gmresir_w64_r64', 'native_gmresir32_w64_r64', 'policy_hetero16_cost_aware'], reps=1, warmup=1, phase='evaluation', tag='E8prof_hard', run_kind='profile', instrument=True, **F))
    return J

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--suite', default='dev'); ap.add_argument('--frozen', default=str(ROOT / 'configs' / 'frozen_config.json')); ap.add_argument('--sizes64', default='8192,16384,24576,32768,36864'); ap.add_argument('--sizes32', default='8192,16384,32768,49152'); ap.add_argument('--e5sizes', default='4096,8192,16384'); ap.add_argument('--e6sizes', default='1024,4096,8192,16384'); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    frozen = json.load(open(a.frozen)) if Path(a.frozen).exists() else dict(nb=256, gmres_m=40, gmres_tol=1e-4, gmres_budget=480, max_outer=30, tau_mult=100)
    plans = dict(dev=lambda: plan_dev(frozen), E2=lambda: plan_e2(frozen), E3=lambda: plan_e3(frozen), E4=lambda: plan_e4(frozen, [int(x) for x in a.sizes64.split(',')], [int(x) for x in a.sizes32.split(',')]), E5=lambda: plan_e5(frozen, [int(x) for x in a.e5sizes.split(',')]), E6=lambda: plan_e6(frozen, [int(x) for x in a.e6sizes.split(',')]), E8=lambda: plan_e8(frozen), E2a=lambda: plan_e2a(frozen))
    if a.cmd == 'plan':
        for s in a.suite.split(','):
            J = plans[s](); (ROOT / 'configs').mkdir(exist_ok=True); json.dump(J, open(ROOT / 'configs' / f'plan_{s}.json', 'w'), indent=0)
            print(s, len(J), 'jobs', sum(len(j['methods']) * (j['reps'] + j['warmup']) for j in J), 'method-runs')
    elif a.cmd == 'run':
        for s in a.suite.split(','):
            J = json.load(open(ROOT / 'configs' / f'plan_{s}.json'))
            with open(ROOT / 'logs' / f'run_{s}.log', 'a') as log: run_jobs(J, RUNS / s, log)
    else: raise SystemExit('unknown cmd')

if __name__ == '__main__': main()
