# Attribution

## Original work in this campaign (new, 2026-11-04)
- Campaign driver `source/src/mpsolve.cu`, `mp_core.cuh`, `mp_kernels.cuh`, `mp_common.cuh`: deterministic GPU matrix generators (structured random orientations with prescribed spectra, RBF kernel matrices, dyadic exact-reference family, Grcar/Wilkinson), fused FP64 residual/componentwise-denominator kernels, double-double residual kernel, row-interchange and conversion kernels, the custom heterogeneous FP32/FP16-tensor-core blocked LU and its FP32 control, the stationary / FGMRES / left-GMRES iterative-refinement engine with recovery policies, the cuSOLVER IRS and MAGMA driver wrappers, metric evaluation, reference computation (FP64 LU + double-double residual refinement), conditioning estimates, timing harness and JSON record emission.
- Independent audit tool `source/src/mpaudit.cu`; orchestration (`campaign.py`), result collection/validation plumbing (`collect.py`), tables/figures/claims (`summarize.py`), packaging (`finalize.py`), the method catalog with precision contracts (`methods_catalog.py`), the Stage-A probe (`src/probe/probe_cusolver.cu`), all documents in this archive.
- E1 scalar-track driver `source/e1_scalar/run_e1.py`, `e1_methods.py` (paper-aligned left-preconditioned GMRES-IR in the operation-rounded model, mpmath 100/120-digit references, metrics), `validate_out.py`.

## Existing project code reused
- `three_precision_lab` (`tpir` package, prior campaign by the same project): operation-rounded FP16/FP32/FP64 LU, triangular solves, arithmetic model, problem generator and metrics; copied unmodified into `source/e1_scalar/three_precision_lab/` and imported by the E1 driver (its 39 tests pass; see tests/). Its own `docs/SOURCES.md` and vendored license apply.
- `mixed_precision_lab_v1_1` (prior Python/PyTorch blocked solver): **not used** for any new measurement; it remains historical/reference evidence in the handoff (`prior_evidence/`). No claim in this archive relies on it.
- Handoff tools (`tools/validate_results.py`, `tools/package_results.py`, `tools/memory_plan.py`) by the project: used for structural validation, packaging and the memory-plan table.

## Vendor / third-party code (not modified)
- NVIDIA CUDA 12.6 toolkit: cuSOLVER 11.7.1 (`cusolverDnXgetrf/Xgetrs`, `cusolverDnIRSXgesv`), cuBLAS 12.6 (`cublasGemmEx`, `Sgemm/Dgemm`, `Strsm`, `Ddot/Dnrm2/Dgemv/Dscal`), CUDA runtime — proprietary NVIDIA licenses.
- MAGMA 2.9.0 (University of Tennessee / UC Berkeley / University of Colorado Denver), BSD-3 license (`build/MAGMA_COPYRIGHT.txt`), built from the official tarball; routines used: `magma_dgetrf_native`, `magma_dgetrf_gpu`, `magma_sgetrf_native`, `magma_[sd]getrs_gpu`, `magma_dxgesv_gmres_gpu`, `magma_htgetrf_gpu`, `magma_sgetrf_recpanel_native` (development comparison only).
- OpenBLAS 0.3.34 (linuxbrew) as MAGMA's host BLAS/LAPACK; OpenSSL libcrypto for SHA-256.
- Python: numpy 2.5.4, scipy 1.18.1, mpmath 1.3.0, matplotlib 3.11.2, pytest 9.2.0.

## Literature
Algorithms follow Carson & Higham (2018), Higham (2019), Haidar et al. (2020), Higham–Pranesh–Zounon (2019), Oktay & Carson (2022) and Amestoy et al. (2024) as mapped in `literature_mapping.md`; no new IR/GMRES algorithm is claimed.

## Agent assistance
All new code, experiments, audits and documents in this archive were produced by the user's coding agent (Claude Fable 5.1, Anthropic) under the user's assignment (`AGENT.md` of the handoff), in a single session on 2026-11-04 with a 4-hour limit; a sub-agent of the same model executed the E1 scalar track. The user supplied the assignment, prior code, prior evidence and the hardware; all measurements were executed on the user's Tesla V100-SXM2-32GB machine. Interpretation beyond the factual run summary is left to the user's independent analysis.
