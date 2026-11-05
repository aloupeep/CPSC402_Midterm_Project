#!/usr/bin/env python3
"""Assemble env/, configs/, source/, build/, tests/, logs/, profiles/, manifest.json for the result package (after collect.py and summarize.py)."""
from __future__ import annotations
import glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path('/home/worker/lab/campaign'); OUT = ROOT / 'writeup_results'; HANDOFF = Path('/home/worker/lab/handoff/mixed_precision_writeup_handoff')
ENV = dict(os.environ, LD_LIBRARY_PATH='/home/worker/lab/campaign/build_magma/magma-2.9.0/lib:/home/linuxbrew/.linuxbrew/lib')
def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120, env=ENV).stdout
    except Exception as e: return f'ERROR {e}'
def cp(src, dst):
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).is_dir(): shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    elif Path(src).exists(): shutil.copyfile(src, dst)
stats = json.load(open(OUT / 'summaries' / 'collect_stats.json'))
# ---- source snapshot
for f in ['src/mp_common.cuh', 'src/mp_kernels.cuh', 'src/mp_core.cuh', 'src/mpsolve.cu', 'src/mpaudit.cu', 'src/probe/probe_cusolver.cu', 'campaign.py', 'collect.py', 'summarize.py', 'audit.py', 'finalize.py', 'methods_catalog.py', 'build/build.sh']: cp(ROOT / f, OUT / 'source' / f)
for f in glob.glob(str(ROOT / 'e1_scalar' / '*.py')) + glob.glob(str(ROOT / 'e1_scalar' / '*.md')): cp(f, OUT / 'source' / 'e1_scalar' / Path(f).name)
cp(ROOT / 'e1_scalar' / 'three_precision_lab', OUT / 'source' / 'e1_scalar' / 'three_precision_lab')
cp(ROOT / 'build_magma' / 'magma-2.9.0' / 'make.inc', OUT / 'build' / 'magma_make.inc')
(OUT / 'source' / 'README.md').write_text('''# Source snapshot
- `src/`: campaign driver (`mpsolve.cu` + headers) and independent audit tool (`mpaudit.cu`), CUDA C++17, built for sm_70 only (see `build/`).
- `campaign.py`: job planning and sequential GPU execution (single benchmark owner via an exclusive file lock, taskset cores 0-7, OPENBLAS_NUM_THREADS=8).
- `collect.py`, `summarize.py`, `finalize.py`: build the result contract, tables, figures, manifest. `audit.py`: regeneration verification + independent residual recomputation.
- `methods_catalog.py`: precision contracts (methods.json).
- `e1_scalar/`: the scalar operation-rounded three-precision track driver (`run_e1.py`, `e1_methods.py`, `validate_out.py`) and an unmodified copy of the prior `three_precision_lab` package it imports.
- MAGMA 2.9.0 was built from the official tarball (sha256 recorded in build/), not vendored here (license: BSD-3, see build/MAGMA_COPYRIGHT.txt).
''')
cp(ROOT / 'build_magma' / 'magma-2.9.0' / 'COPYRIGHT', OUT / 'build' / 'MAGMA_COPYRIGHT.txt')
# ---- build
for f in ['build/build.sh', 'build/build_command.txt', 'build/build_log.txt', 'build/mpsolve.sha256', 'build/mpaudit.sha256', 'build/build_command_audit.txt', 'build/nvcc_version.txt']: cp(ROOT / f, OUT / f)
magma_log = ROOT / 'logs' / 'magma_build.log'
if magma_log.exists(): (OUT / 'build' / 'magma_build_log_tail.txt').write_text(magma_log.read_text()[-20000:])
(OUT / 'build' / 'magma_tarball.sha256').write_text(sh(f"sha256sum {ROOT}/build_magma/magma-2.9.0.tar.gz"))
(OUT / 'build' / 'BUILD_HISTORY.md').write_text('''# Build history (binaries used during the campaign)
| time (UTC-0) | binary sha256 | change | used for |
|---|---|---|---|
| 21:04 | (first build) | initial driver | smoke tests only |
| 21:08-21:16 | 0465b711f951cc6d... (21:16) | GMRES scratch fix, vendor RHS copies, B integrity check, factor-reuse flag | development tuning (runs/dev) |
| 21:20 | see build/mpsolve.sha256 | added optional MAGMA recursive-panel path (`--panel magma`, NOT used by any frozen method; default `cusolver`) | all evaluation/confirmation suites (E2-E8) and dev_panel comparison |

The driver source in `source/src/` is the final version. Trials carry `source_sha256` = hash of this snapshot; development-phase trials were produced by the 21:16 binary whose only difference is the absence of the unused `--panel magma` code path and the `panel` field in records. The audit tool `mpaudit` is a separate executable built from the same headers (hash in build/mpaudit.sha256).
MAGMA 2.9.0 was built from source with `make lib -j72`, GPU_TARGET=Volta (sm_70 only), gcc 13.3, nvcc 12.6.85, OpenBLAS 0.3.34 (linuxbrew) for host BLAS/LAPACK; see magma_make.inc and magma_build_log_tail.txt. Build time about 10 minutes on 80 cores.
Compiler flags for the driver: `-O3 -std=c++17 -arch=sm_70 -lineinfo -DHAVE_MAGMA` (no fast-math; FMA contraction at nvcc default; explicit __fma_rn/__dmul_rn in the generator kernels).

## Later rebuilds during the campaign (chronological notes)
''' + ((ROOT / 'build' / 'BUILD_HISTORY_NOTE.txt').read_text() if (ROOT / 'build' / 'BUILD_HISTORY_NOTE.txt').exists() else ''))
# ---- env / capabilities
caps = {}
try: caps = json.loads(sh(f"{ROOT}/build/mpsolve caps"))
except Exception as e: caps = dict(error=str(e))
probe = [json.loads(l) for l in open(ROOT / 'logs' / 'probe_cusolver_2048_v2.jsonl') if l.startswith('{')] if (ROOT / 'logs' / 'probe_cusolver_2048_v2.jsonl').exists() else []
cap = dict(collected_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), device=caps, nvidia_smi_start=(ROOT / 'env' / 'nvidia_smi_query_start.csv').read_text() if (ROOT / 'env' / 'nvidia_smi_query_start.csv').exists() else None, nvidia_smi_end=sh('nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.sm,clocks.max.sm,clocks.mem,power.limit,power.draw,temperature.gpu,ecc.mode.current,pstate,compute_mode,persistence_mode --format=csv'),
           cuda_toolkit='12.6 (nvcc V12.6.85)', cusolver='11.7.1.2', cublas='12.6.4.1', driver='570.133.20', magma='2.9.0 built from source (sm_70)', cpu='Intel Xeon Gold 6138 @ 2.00GHz, 80 logical CPUs; benchmark process pinned to cores 0-7 with taskset; OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=8', host_memory_gb=503, os='Linux 5.15.0-139-generic (Ubuntu 24.04 userland, gcc 13.3.0)', platform='Dell PowerEdge C4140',
           python=dict(venv='/home/worker/lab/venv', python='3.14.7', numpy='2.5.4', scipy='1.18.1', mpmath='1.3.0', torch='2.10.0+cu126 (present, NOT used by any measurement)', matplotlib='3.11.2'),
           loaded_libraries=(ROOT / 'env' / 'ldconfig_cuda.txt').read_text() if (ROOT / 'env' / 'ldconfig_cuda.txt').exists() else None, ldd_mpsolve=sh(f'ldd {ROOT}/build/mpsolve | grep -E "cusolver|cublas|magma|openblas|cudart"'),
           clock_control='GPU clocks could not be locked (nvidia-smi -lgc requires privileges not used); boost clocks vary with load; idle SM clock 1012 MHz, max 1530 MHz; ECC enabled; persistence mode enabled',
           backend_capability_table=dict(
               cusolver_native_lu_fp32_fp64='available and used (cusolverDnXgetrf/Xgetrs 64-bit API)',
               cusolver_irs_main64_lowest16_classical='available (probe + campaign); fallback observable only via sign of niters',
               cusolver_irs_main64_lowest16_gmres='available for nrhs=1 only (status 21 for nrhs>1)', cusolver_irs_main64_lowest32='available', cusolver_irs_main32_lowest16='available', cusolver_irs_lowest_bf16_tf32='rejected on sm_70 (bufferSize status 4): not supported, no claims made',
               magma_native_lu_fp32_fp64='available (magma_[sd]getrf_native)', magma_hybrid_lu='available (CPU panels via OpenBLAS, 8 threads; slower than native here)', magma_dxgesv_gmres_gpu_SHT_IRSTRS='available; produced NaN while reporting convergence on one case (recorded)', magma_dxgesv_gmres_gpu_SHT_GMSTRS='available for nrhs=1; returns zeros for nrhs>1 (recorded as unsupported)', magma_htgetrf_gpu='available; FP32 factors with FP16-TC trailing updates; for n<=block size it is an FP32 factorization',
               custom_hetero_fp16tc_lu='implemented (FP32 panel via cuSOLVER, FP16-operand tensor-core trailing update via cublasGemmEx, FP32 accumulate)', custom_fp32_blocked_control='implemented (same code path with cublasSgemm)', fgmres_ir='implemented on device (MGS, FP64 basis)', left_paper_gmres='implemented (moderate n, FP64 application via promoted copy)',
               nsys='not installed (profiling evidence = instrumented stage timers and solver counters; ncu present but not used)', torch_pytorch='not used'),
           unknowns=['cuSOLVER IRS internal factorization/solve/residual precisions and stopping rule', 'MAGMA kernel-level arithmetic inside xshgetrf/htgetrf beyond the source-level structure read', 'exact cuSOLVER Xgetrf internal blocking', 'GPU clock state during each repetition (not locked)'],
           probe_records=probe, frozen_config_sha256=stats['frozen_sha256'], source_sha256=stats['source_sha256'])
json.dump(cap, open(OUT / 'env' / 'capabilities.json', 'w'), indent=1)
for f in ['nvidia_smi_query_start.csv', 'nvidia_smi_full_start.txt', 'lscpu.txt', 'meminfo.txt', 'uname.txt', 'nvcc_version.txt', 'gcc_version.txt', 'ldconfig_cuda.txt']: cp(ROOT / 'env' / f, OUT / 'env' / f)
(OUT / 'env' / 'pip_freeze.txt').write_text(sh('/home/worker/lab/venv/bin/pip freeze 2>/dev/null'))
(OUT / 'env' / 'nvidia_smi_full_end.txt').write_text(sh('nvidia-smi -q'))
(OUT / 'env' / 'memory_plan_tool_output.txt').write_text(sh(f'cd {HANDOFF} && python3 tools/memory_plan.py --n 8192 12288 16384 24576 32768 36864 49152 --bytes-per-element 24 --workspace-gib 2'))
# ---- configs
for f in glob.glob(str(ROOT / 'configs' / '*')): cp(f, OUT / 'configs' / Path(f).name)
cp(HANDOFF / 'configs' / 'campaign_plan.json', OUT / 'configs' / 'requested_campaign_plan.json')
# ---- tests
for f in glob.glob(str(ROOT / 'logs' / 'probe_cusolver_*.jsonl')) + [str(ROOT / 'logs' / 'handoff_tools_unittest.log')]: cp(f, OUT / 'tests' / Path(f).name)
cp(ROOT / 'e1_scalar' / 'tests' / 'pytest_log.txt', OUT / 'tests' / 'e1_three_precision_lab_pytest.txt')
for d in ['smoke2', 'smoke4', 'dev_panel']:
    for f in glob.glob(str(ROOT / 'runs' / d / '*trials*.jsonl')) + glob.glob(str(ROOT / 'runs' / d / '*/trials*.jsonl')): cp(f, OUT / 'tests' / f'{d}__{Path(f).name}')
(OUT / 'tests' / 'generator_selftest.txt').write_text(sh(f'{ROOT}/build/mpsolve selftest --n 256 --family prescribed --spectrum identity --seed 9100') + sh(f'{ROOT}/build/mpsolve selftest --n 384 --family prescribed --spectrum identity --seed 9101'))
cp(ROOT / 'audit', OUT / 'tests' / 'audit')
cp(ROOT / 'e0_replay', OUT / 'tests' / 'e0_teaching_case_replay')
(OUT / 'tests' / 'README.md').write_text('''# Tests and checks
- `probe_cusolver_*.jsonl`: Stage A native capability probe (cuSOLVER LU, IRS modes incl. rejected BF16/TF32, FP16-input GEMM dispatch). The v1 file shows the CUBLAS_TENSOR_OP_MATH trap (SGEMM silently down-converted to FP16, 50 TFLOPS); v2 uses default math mode.
- `generator_selftest.txt`: orthogonality (max deviation ~1.5e-15) and density (no zero entries) of the structured random orientation generator.
- `smoke2__*`, `smoke4__*`: per-method smoke tests at n=512/1024 (all method families, both modes, repeatability across reps, B-integrity check).
- `dev_panel__*`: MAGMA recursive panel vs cuSOLVER panel comparison (not adopted).
- `handoff_tools_unittest.log`: the handoff's 18 structural tool tests (pass).
- `e1_three_precision_lab_pytest.txt`: prior scalar-model package tests (39 pass) used by the E1 track.
- `e0_teaching_case_replay/`: E0 replay of the prior teaching case cb384647987dbe4e saved iterates against independent numpy-FP64 / mpmath-60-digit stored-system residuals and the stored reference (reproduces the prior forward errors 3.0261e-8 (hsd) and 1.6535e-5 (hss)).
- `audit/`: regeneration verification (fresh process) and independent FP64 + double-double residual recomputation of every saved solution; `host_crosscheck.json` = numpy float64/long-double cross-check on embedded (n<=512) cases.
''')
# ---- logs
for f in glob.glob(str(ROOT / 'logs' / '*')): cp(f, OUT / 'logs' / Path(f).name)
for f in glob.glob(str(ROOT / 'runs' / '*' / '*' / 'done_*.json')):
    p = Path(f); cp(f, OUT / 'logs' / 'jobs' / p.parent.parent.name / p.parent.name / p.name)
# ---- profiles: instrumented E8 records + inner histories
prof = [json.loads(l) for l in open(OUT / 'trials.jsonl') if '"run_kind": "profile"' in l]
json.dump(prof, open(OUT / 'profiles' / 'e8_instrumented_trials.json', 'w'), indent=1)
pids = {t['trial_id'] for t in prof}
with open(OUT / 'profiles' / 'e8_histories.jsonl', 'w') as f:
    for l in open(OUT / 'histories.jsonl'):
        if json.loads(l)['trial_id'] in pids: f.write(l)
(OUT / 'profiles' / 'README.md').write_text('Instrumented (run_kind=profile) trials: sequential stage timers (factorization incl. conversion/scaling, initial solve, refinement, convergence checks), solver counters and full outer/inner histories. No nsys traces (nsys not installed); kernel-level profiler output is not available in this archive. Stage timers are host-synchronized wall clocks and do not overlap.\n')
# ---- manifest
limits = ['see LIMITATIONS.md']
man = dict(schema_version='1.0', campaign_id='mixed_precision_lu_writeup_extension_2026-11-04_v100sxm2', is_example=False, source_sha256=stats['source_sha256'], frozen_config_sha256=stats['frozen_sha256'], environment_file='env/capabilities.json', completed=False, created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           n_cases=stats['n_cases'], n_trials=stats['n_trials'], limitations_and_sidecars=['no sidecar matrices: n<=512 embedded (matrices/), larger cases regenerable from matrices/recipes with fresh-process hash verification (tests/audit/audit_results.json)', 'completed=false: the time-limited plan trimmed seeds/sizes/repetitions at large n and omitted some planned combinations; see RUN_SUMMARY.md and coverage.csv', 'E1 scalar track records are in trials.jsonl with suite=E1 (CPU model, no performance claims)'],
           sidecars=[], validator='tools/validate_results.py from the handoff')
json.dump(man, open(OUT / 'manifest.json', 'w'), indent=1)
print('finalize done')
