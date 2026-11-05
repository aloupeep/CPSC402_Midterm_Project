# mixed_precision_campaign_code — source, scripts and environment of the 2026-11-04 campaign

This archive contains everything that produced `mixed_precision_writeup_results.zip`
(sha256 e0f56efa6bc7537737d2e64863d56f9fb7a01da1e2c3e27f899a18efc296550a) **except the
results themselves**. It is a snapshot of `/home/worker/lab/campaign/` on the V100 box with
the run outputs removed (see "Excluded" below). `ENVIRONMENT.md` lists every requirement
(hardware, OS, toolchain, libraries, MAGMA build, Python packages) and the exact command
sequence that was used.

## Contents

| Path | What it is |
|---|---|
| `ENVIRONMENT.md` | Required environment, build and run instructions, hard-coded paths. |
| `src/mpsolve.cu`, `src/mp_core.cuh`, `src/mp_kernels.cuh`, `src/mp_common.cuh` | The CUDA/C++ campaign driver: deterministic matrix generators, cuSOLVER / MAGMA / custom heterogeneous FP32-panel + FP16-tensor-core LU, stationary IR, FGMRES-IR, left GMRES-IR, recovery policies, cuSOLVER IRS and MAGMA expert-driver wrappers, fused FP64 and double-double residual metrics, timing and memory accounting, JSON trial records. |
| `src/mpaudit.cu` | Independent audit tool: regenerates a stored case in a fresh process, verifies hashes, recomputes FP64 / double-double metrics of saved solutions, dumps small matrices. |
| `src/probe/probe_cusolver.cu` | Stage-A capability probe (cuSOLVER LU, IRS modes, FP16 GEMM dispatch). Its output is in `logs/probe_cusolver_2048*.jsonl`. |
| `build/` | `build.sh` (exact nvcc command), build logs, `BUILD_HISTORY_NOTE.txt` (the two rebuilds during the campaign and what they changed), the final binaries `mpsolve` / `mpaudit` / `src/probe/probe_cusolver` and their SHA-256 hashes (binaries are sm_70, linked against the paths in ENVIRONMENT.md). |
| `magma/make.inc`, `magma/magma-2.9.0.tar.gz.sha256`, `logs/magma_build.log` | MAGMA 2.9.0 build configuration and full build log (the MAGMA source tree and tarball are not included; see ENVIRONMENT.md). |
| `campaign.py` | Job planner and sequential GPU runner for suites `dev, E2, E2a, E3, E4, E5, E6, E8` (GPU lock, done markers, per-case directories under `runs/`). |
| `methods_catalog.py` | The 57 GPU-track methods with their precision contracts (`methods.json`). |
| `collect.py`, `summarize.py`, `dev_table.py`, `audit.py`, `audit_summary.py`, `write_run_summary.py`, `finalize.py`, `report_numbers.py`, `make_package.sh`, `upload_results.sh` | Post-processing: raw run directories -> RESULTS_CONTRACT files -> tables/figures/claims -> RUN_SUMMARY.md -> manifest/provenance -> validation -> zip -> upload. |
| `configs/` | `frozen_config.json` (+ sha256 and amendment history) and the exact job plans that were executed (`plan_<suite>.json`, including the extra separate-process jobs appended for n=24576/36864). |
| `docs_static/` | LIMITATIONS.md, ATTRIBUTION.md, literature_mapping.md, README.md as shipped inside the results archive. |
| `e1_scalar/` | CPU three-precision scalar track (E1): `run_e1.py`, `e1_methods.py` (mpmath references, GMRES-IR on the prior package's FP16 factors), `validate_out.py`, `CHANGES.md`, and an unmodified copy of the handoff's `three_precision_lab` package with its pytest log. The E1 outputs (`e1_scalar/out/`) are excluded. |
| `e0_replay/` | Replay of the prior teaching case against independent residual evaluation (script, note, and its small JSON output). |
| `env/` | Machine capture at campaign start (nvidia-smi, lscpu, meminfo, uname, nvcc, gcc, ldconfig) plus `pip_freeze_venv.txt` and `pip_freeze_system_python3.txt`. |
| `logs/` | MAGMA build log, handoff tool unit-test log, probe outputs. Per-suite run logs are excluded (they are in the results archive). |
| `SHA256SUMS` | Hashes of every file in this archive. |

## Excluded on purpose (result artifacts)

`runs/` (1.3 GB raw per-case records, saved X/B/Xref arrays, histories), `writeup_results/`
and the packaged results zip, `audit/` (audit_results.json, host_crosscheck.json),
`e1_scalar/out/`, `logs/run_*.log` and orchestrator outputs, the MAGMA source tree and
tarball (1.4 GB; reproducible from the pinned tarball + `magma/make.inc`), the Python venv,
`__pycache__`, and the handoff archive itself (`mixed_precision_writeup_handoff.zip`,
sha256 f43c8239807d82297d9d257fa172cd022928baecfd43e3239b8ef9e3699e084d), which the
post-processing scripts read for `tools/validate_results.py`, `tools/package_results.py`,
`prior_code/three_precision_lab` and `prior_evidence/teaching_case`.

## Quick start

1. Read `ENVIRONMENT.md`, install the toolchain, unpack MAGMA 2.9.0 next to this tree as
   `build_magma/magma-2.9.0`, copy `magma/make.inc` into it and run `make lib -j72`.
2. `bash build/build.sh` (edit the absolute paths at the top if the tree is not at
   `/home/worker/lab/campaign`), then `build/mpsolve caps` and `build/mpsolve selftest --n 4096`.
3. `python3 campaign.py plan --suite dev,E2,E6,E4,E2a,E3,E5,E8` and
   `python3 campaign.py run --suite dev`, then the evaluation suites (hours of GPU time; see
   ENVIRONMENT.md section 6 for the exact order, sizes and the E1 / E0 CPU steps).
4. `bash make_package.sh` to audit, collect, summarize, validate and package.
