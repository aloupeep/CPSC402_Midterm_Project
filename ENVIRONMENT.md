# ENVIRONMENT.md — required environment for the mixed-precision LU / IR campaign

Everything below is what was actually used on 2026-11-04 (captured in `env/`). Items marked
"required" are needed to build and run; items marked "as used" are what was present and can
be substituted with care.

## 1. Hardware (as used)

| Component | Value |
|---|---|
| GPU | 1x NVIDIA Tesla V100-SXM2-32GB, compute capability 7.0 (`sm_70`), 32768 MiB, ECC on, persistence mode on, power limit 300 W, boost clocks not locked (SM 1530 MHz max) |
| Driver | 570.133.20 (reports CUDA 12.8 capability; toolkit used is 12.6) |
| CPU | 2x Intel Xeon Gold 6138 @ 2.00 GHz, 40 cores / 80 threads, AVX-512 |
| RAM | 503 GiB (MemTotal 527,696,072 kB) |
| Core pinning | GPU driver, audit and MAGMA host work on cores 0-7 (`taskset -c 0-7`, 8 OpenBLAS threads); the E1 CPU track on cores 8-79 (54 single-threaded worker processes) |

The largest cases (FP64 n=36,864: 10.1 GiB for A; FP32 n=49,152: 9.06 GiB) need a 32 GB
card. The driver admits a job only if its estimated footprint fits in (free memory - 512 MiB),
counting buffers that are already allocated and reusable, and records `memory_skip`
otherwise. A GPU with a different compute capability needs `-arch=` in `build/build.sh` and
`GPU_TARGET` in `magma/make.inc` changed; the FP16 tensor-core paths (`hetero16_*`,
`custom*`, cuSOLVER IRS `l16`, MAGMA `htgetrf`) need tensor cores (sm_70+). On sm_70 the
BF16 / TF32 IRS modes are rejected by cuSOLVER; that is recorded, not worked around.

## 2. Operating system and toolchain

| Item | Value | Status |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (container), kernel 5.15.0-139-generic, glibc 2.39, x86_64 | as used |
| Host C/C++ compiler | gcc / g++ 13.3.0 (Ubuntu, `/usr/bin/gcc`) — used by nvcc as host compiler and by MAGMA for C/C++ | required (any gcc nvcc 12.6 supports) |
| CUDA toolkit | 12.6 at `/usr/local/cuda` (nvcc V12.6.85); cuBLAS / cuBLASLt 12.6.4.1, cuSOLVER 11.7.1.2, cuSPARSE and cudart from the same toolkit | required (cuSOLVER >= 11.7 for the 64-bit `cusolverDnXgetrf` and `cusolverDnIRSXgesv` APIs that the driver uses) |
| GNU make | 4.3 (`/usr/bin/make`) | required for MAGMA |
| Fortran | gfortran 16.2.0 from Linuxbrew (`/home/linuxbrew/.linuxbrew/bin/gfortran`) — only MAGMA's Fortran sources; libgfortran.so.5 is then a runtime dependency of libmagma.so | required for MAGMA build |
| OpenBLAS | 0.3.34 from Linuxbrew (`/home/linuxbrew/.linuxbrew/lib/libopenblas.so.0`), host BLAS/LAPACK for MAGMA and for the driver's host-side pieces (`-lopenblas`) | required |
| OpenSSL | headers from Ubuntu `libssl-dev` 3.0.13 (`/usr/include/openssl/evp.h`); at link/run time `-lcrypto` resolved to Linuxbrew `libcrypto.so.3` (OpenSSL 3.6.3). Used only for SHA-256 of matrices / RHS / solutions via EVP | required (any OpenSSL 1.1+/3.x) |
| libgomp | OpenMP runtime (`-fopenmp`; resolved to Linuxbrew libgomp.so.1 at run time) | required |
| Linuxbrew prefix | `/home/linuxbrew/.linuxbrew` (also provides `python3` 3.14.7 and cmake 4.4.2; cmake is not used — MAGMA is built with its Makefile) | as used |

`ldd build/mpsolve` on the box resolves: libmagma.so (project build), libcusolver.so.11,
libcublas.so.12, libcublasLt.so.12, libcusparse.so.12, libcudart.so.12 (all `/usr/local/cuda/lib64`),
libcrypto.so.3, libopenblas.so.0, libgomp.so.1 (Linuxbrew), libgfortran.so.5 (Linuxbrew gcc).

## 3. Python

Two interpreters were used, both CPython 3.14.7:

| Interpreter | Packages that matter | Used by |
|---|---|---|
| `/home/worker/lab/venv/bin/python` (the venv provided with the box) | numpy 2.5.4, scipy 1.18.1, mpmath 1.3.0, matplotlib 3.11.2, pytest 9.2.0 (pytest was the only package installed during the campaign). torch 2.10.0+cu126 and the `nvidia-*-cu12` wheels are present but **not used by any measurement**. Full list: `env/pip_freeze_venv.txt`. | `audit.py`, `summarize.py` (matplotlib figures), `e1_scalar/run_e1.py`, `e1_scalar/validate_out.py`, pytest of `three_precision_lab` |
| `python3` = Linuxbrew 3.14.7 | numpy 2.5.2, mpmath 1.4.1 (Linuxbrew formulae); everything else is stdlib. Full list: `env/pip_freeze_system_python3.txt`. | `campaign.py`, `collect.py`, `dev_table.py`, `audit_summary.py`, `write_run_summary.py`, `finalize.py`, `report_numbers.py`, `e0_replay/replay_teaching_case.py`, the handoff's `validate_results.py` / `package_results.py` |

A single environment with `numpy scipy mpmath matplotlib pytest` satisfies both roles. Nothing
depends on torch or on the pip CUDA wheels; the GPU code is compiled C++/CUDA only.

## 4. MAGMA 2.9.0 (built from source, not installed system-wide)

- Source: `magma-2.9.0.tar.gz` from ICL (https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.9.0.tar.gz),
  sha256 `ff77fd3726b3dfec3bfb55790b06480aa5cc384396c2db35c56fdae4a82c641c` (`magma/magma-2.9.0.tar.gz.sha256`).
- Unpack to `<campaign>/build_magma/magma-2.9.0`, copy `magma/make.inc` into that directory
  (it is `make.inc-examples/make.inc.openblas` reduced to: `BACKEND=cuda`, `OPENBLASDIR=/home/linuxbrew/.linuxbrew`,
  `CUDADIR=/usr/local/cuda`, `FORT=/home/linuxbrew/.linuxbrew/bin/gfortran`, `GPU_TARGET=Volta`, `-fPIC`, `-O3 -DNDEBUG -DADD_ -fopenmp`), then:

      cd build_magma/magma-2.9.0 && make lib -j72 > ../../logs/magma_build.log 2>&1

  This took about 10 minutes on 80 threads and produced `lib/libmagma.a` and `lib/libmagma.so`
  (the sparse library and testers were not built; `make install` was not run). The driver
  compiles against `build_magma/magma-2.9.0/include` and links `-L build_magma/magma-2.9.0/lib -lmagma`
  with an RPATH to that directory. The full build log is `logs/magma_build.log` (deprecation
  warnings only; exit 0).
- MAGMA routines used: `magma_dgetrf_native`, `magma_sgetrf_native`, `magma_dgetrf_gpu` (hybrid),
  `magma_sgetrf_recpanel_native` (panel for the custom LU when `--panel magma`), `magma_dxgesv_gmres_gpu`
  (SHT/SS x IRSTRS/GMSTRS expert drivers), `magma_htgetrf_gpu`, `magma_dgetrs_gpu`/`magma_sgetrs_gpu`, `magma_dlaswp`.

## 5. Building the campaign binaries

- Driver: `bash build/build.sh` -> `build/mpsolve`. Exact command (also in `build/build_command.txt`):

      nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo -DHAVE_MAGMA -I<MAGMA>/include -I/usr/local/cuda/include \
           -o build/mpsolve src/mpsolve.cu \
           -L<MAGMA>/lib -lmagma -lcusolver -lcublas -lcublasLt -lcusparse -lcudart -lcrypto \
           -L/home/linuxbrew/.linuxbrew/lib -lopenblas \
           -Xlinker -rpath=<MAGMA>/lib -Xlinker -rpath=/home/linuxbrew/.linuxbrew/lib \
           -Xcompiler -fopenmp -Xlinker --disable-new-dtags

  `--disable-new-dtags` makes the rpath a `DT_RPATH` (searched before `LD_LIBRARY_PATH`) so the
  Linuxbrew `libopenblas.so.0` is found by the transitive MAGMA dependency. The final binary's
  sha256 is `ae10e3e0dfc7ee317eb56917f18b0bb4b4bcdb03128b15b9005d9166dff425e8` (`build/mpsolve.sha256`);
  `build/BUILD_HISTORY_NOTE.txt` documents the two rebuilds during the campaign (policy ladder
  fix at 22:16, admission-check fix at 22:59) and which suites ran with which binary.
- Audit tool: from `build/` with `MAGMA=<campaign>/build_magma/magma-2.9.0`:

      nvcc -O3 -std=c++17 -arch=sm_70 -DHAVE_MAGMA -I$MAGMA/include -o mpaudit ../src/mpaudit.cu \
           -L$MAGMA/lib -lmagma -lcusolver -lcublas -lcublasLt -lcusparse -lcudart -lcrypto \
           -L/home/linuxbrew/.linuxbrew/lib -lopenblas

  (sha256 `45f9d4c74d187b13d68df6b7f31d75a4256d44362d016a2db7eee97edf1ba093`). It has no RPATH, so run it with
  `LD_LIBRARY_PATH` set as in section 6.
- Capability probe (optional): `cd src/probe && nvcc -O2 -arch=sm_70 -o probe_cusolver probe_cusolver.cu -lcusolver -lcublas && ./probe_cusolver 2048`.
- Compile warnings: one unused-variable warning in `mp_core.cuh` (see `build/build_log.txt`); nothing else.

## 6. Runtime environment and command sequence

Environment variables (set by `campaign.py` and `audit.py` for every subprocess):

    LD_LIBRARY_PATH=<campaign>/build_magma/magma-2.9.0/lib:/home/linuxbrew/.linuxbrew/lib:$LD_LIBRARY_PATH
    OPENBLAS_NUM_THREADS=8  OMP_NUM_THREADS=8  MKL_NUM_THREADS=8
    taskset -c 0-7 <command>

The E1 CPU track sets `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` per worker.
The GPU must be otherwise idle; `campaign.py` serializes GPU jobs through `<campaign>/gpu.lock`
(fcntl) so several orchestrators can be started without overlapping GPU work (this is how the
extra n=24,576 / 36,864 separate-process jobs were appended while a queue was running).

Order used on 2026-11-04 (wall-clock in parentheses):

1. `env/` capture; MAGMA build (20:41-20:51); `src/probe` probe; `bash build/build.sh`;
   `build/mpsolve caps`; `build/mpsolve selftest --n <n>` (generator orthogonality / density check).
2. Plans: `python3 campaign.py plan --suite dev,E2,E6,E4,E2a,E3,E5,E8 --sizes64 8192,12288,16384,24576,32768,36864 --sizes32 8192,16384,32768,49152 --e5sizes 4096,8192,16384 --e6sizes 1024,4096,8192,16384`
   (the shipped `configs/plan_*.json` are the versions actually executed; E4/E5 were re-planned
   with reduced FP32 sizes and extra jobs were appended by hand, see `configs/FROZEN_CONFIG_HISTORY.md`
   and `docs_static/LIMITATIONS.md`).
3. Development data: `python3 campaign.py run --suite dev` (seeds 9100/9101; 21:16-21:19), then the
   frozen config was fixed (`configs/frozen_config.json`, sha256 in `frozen_config.sha256`).
4. Evaluation queue: `python3 campaign.py run --suite E2,E6,E4,E2a,E3,E5,E8` (21:21-23:38, with
   E6 re-run after the 22:16 rebuild and the n=36,864 FP64 baselines re-queued after the 22:59
   rebuild). Each case lives in `runs/<suite>/<case_id>/` (`case_*.json`, `trials_*.jsonl`,
   `hist_*.jsonl`, `X_*`, `B_*`, `Xref_*`, `done_*.json`). Pre-emptive time-budget skips were
   written as done markers with rc=-2 (`skipped_time_budget`).
5. E1 (CPU, concurrently with the GPU queue):
   `taskset -c 8-79 /home/worker/lab/venv/bin/python e1_scalar/run_e1.py --out e1_scalar/out --procs 54`
   (about 4 minutes), then `e1_scalar/validate_out.py`; `pytest` in `e1_scalar/three_precision_lab`
   (39 tests, `e1_scalar/tests/pytest_log.txt`).
6. E0: `python3 e0_replay/replay_teaching_case.py` (reads the handoff's `prior_evidence/teaching_case`).
7. Handoff tool unit tests: `python3 -m unittest -v` in `<handoff>/tests` (`logs/handoff_tools_unittest.log`).
8. Packaging: `bash make_package.sh` = `audit.py` (fresh-process regeneration + FP64/double-double
   re-evaluation of every saved solution; about 20 minutes) -> `collect.py` -> `summarize.py` ->
   `dev_table.py` -> `audit_summary.py` -> copy `docs_static/*` -> `write_run_summary.py` ->
   `finalize.py` (manifest, provenance, source copy, env capture) -> handoff `validate_results.py`
   -> handoff `package_results.py` -> `/home/worker/lab/mixed_precision_writeup_results.zip`.
   `report_numbers.py` prints the numbers quoted in RUN_SUMMARY.md.
9. `bash upload_results.sh <zip>` uploads to gofile.io (anonymous API) and prints the download page.

Driver CLI: `build/mpsolve run|regen|caps|selftest --key value ...` (keys are generated by
`campaign.py:driver_cmd()`: `--n --nrhs --family --spectrum --kappa --seed --dtype --methods
--reps --warmup --tag --phase --run_kind --host --nb --gmres_m --gmres_tol --gmres_budget
--max_outer --tau_mult --outdir ...`; read `driver_cmd()` and `parse_args()` in `src/mpsolve.cu`
for the full list). `build/mpaudit --case <case_*.json> --xlist <file> [--xref ..] [--xexact ..] [--dump-a ..]`.

## 7. Hard-coded paths (change these to relocate)

| Path | Where |
|---|---|
| `/home/worker/lab/campaign` (ROOT) | `campaign.py`, `audit.py`, `audit_summary.py`, `collect.py`, `finalize.py`, `summarize.py`, `write_run_summary.py`, `dev_table.py`, `report_numbers.py`, `make_package.sh`, `e0_replay/replay_teaching_case.py`, `e1_scalar/run_e1.py` (defaults only), `e1_scalar/validate_out.py`, `build/build.sh` (`SRC`, `MAGMA`, `OUT`) |
| `/home/worker/lab/campaign/build_magma/magma-2.9.0/lib` and `/home/linuxbrew/.linuxbrew/lib` | `LD_LIBRARY_PATH` in `campaign.py`, `audit.py`, `finalize.py`; rpath in `build/build.sh` |
| `/home/worker/lab/venv` | `make_package.sh`, `finalize.py`, `e1_scalar/run_e1.py` docstring |
| `/home/worker/lab/handoff/mixed_precision_writeup_handoff` | `make_package.sh` (tools), `finalize.py` (provenance), `e0_replay/replay_teaching_case.py` (teaching case) |
| `/home/worker/lab/mixed_precision_writeup_results.zip` | `make_package.sh`, `upload_results.sh` default |

## 8. Inputs that are not in this archive

- `mixed_precision_writeup_handoff.zip` (sha256 `f43c8239807d82297d9d257fa172cd022928baecfd43e3239b8ef9e3699e084d`),
  extracted to `/home/worker/lab/handoff/mixed_precision_writeup_handoff`: specifications
  (`docs/`), `tools/validate_results.py`, `tools/package_results.py`, `tools/memory_plan.py`,
  `prior_code/three_precision_lab` (copied unmodified into `e1_scalar/`), `prior_evidence/teaching_case`.
- MAGMA 2.9.0 tarball (section 4).
- All run outputs: `runs/`, `audit/`, `e1_scalar/out/`, `writeup_results/`, run logs, and the
  packaged results `mixed_precision_writeup_results.zip`
  (sha256 `e0f56efa6bc7537737d2e64863d56f9fb7a01da1e2c3e27f899a18efc296550a`, 424,914,869 bytes).

## 9. Behaviours worth knowing before re-running

- Never set `CUBLAS_TENSOR_OP_MATH` on the shared cuBLAS handle: it silently down-converts
  SGEMM operands to FP16 (observed 50 TFLOPS "FP32"). The driver uses the default math mode and
  requests FP16 tensor cores only through explicit `cublasGemmEx` FP16-operand calls.
- cuSOLVER IRS `infos` objects cannot be reused across calls (status 2); the wrapper creates
  fresh params/infos per solve. IRS fallback to the FP64 factorization is only visible through
  the sign of `niters`; the GMRES IRS modes accept `nrhs=1` only; BF16/TF32 are rejected on sm_70.
- MAGMA `dxgesv_gmres_gpu` returns an all-zero solution for `nrhs>1` (recorded as unsupported);
  the SHT/SS classical IR variants can return NaN while reporting convergence (recorded as
  nonfinite failures). MAGMA's first call is slow if OpenBLAS threads compete with other CPU
  work, hence the core pinning above.
- The custom blocked LU needs `n/32+16` ints of device info scratch for the recursive panel;
  the admission check must subtract already-allocated reusable buffers (both fixed in this source).
- `campaign.py run` is resumable: finished jobs have `done_*.json` markers and are skipped.
