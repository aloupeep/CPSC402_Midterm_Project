# Mixed-precision LU: the missing three-precision comparisons

This is a **standalone addition**, not a replacement for `mixed_precision_lab_v1_1`.
It studies standard LU-based iterative refinement with explicit factorization,
working, and residual precision. The central method is `(FP16, FP32, FP64)`.
The project remains **Mixed-Precision LU and Iterative Refinement**, not a GPU case study.

## What is actually new

The core implementation performs partial-pivoted LU **and both triangular solves**
with an explicitly rounded binary16 arithmetic model. Every division, product,
and subtraction is rounded, including the products inside elimination. The
working solution and every update are FP32; residuals are FP64. This is not
`A.astype(float16).astype(float32)` followed by an FP32 solver.

The pure-half numerical model runs on a CPU. This avoids relying on an unsupported
half-precision LU library interface and makes every arithmetic role inspectable.
**Its Python/NumPy runtime is not an FP16 hardware benchmark.**

The optional CUDA bridge separately adapts the previously tested mixed blocked
factorization to **FP32 working updates**. It remains FP32 panels, FP32 triangular
solves, FP16-input/FP32-output trailing products—not pure FP16 LU.

## Dependencies and environment

Core: Python 3.10+, NumPy, SciPy, Matplotlib. `mpmath` is needed only for the
independent high-precision audit. PyTorch is needed only for the optional bridge.
Keep your existing working environment. No compiler, CUDA toolkit installation,
CuPy, or MAGMA is required. Do **not** upgrade your working V100 PyTorch environment
just to run this package.

```bash
unzip three_precision_lab.zip
cd three_precision_lab
# Activate your existing Python environment here.
python -m tpir doctor
python -m unittest discover -s tests -v
```

No `pip install .` is necessary; invoke from the extracted directory. Install only
missing dependencies. The automated script does not install or upgrade packages.

## One command: complete numerical comparison, optionally plus GPU bridge

```bash
# CPU numerical investigation, works on the Mac or on the GPU machine's CPU.
bash scripts/run_all.sh results/your_machine

# Same investigation plus the native/mixed blocked CUDA comparison:
bash scripts/run_all.sh results/your_machine --cuda
```

Upload **`results/your_machine_results.zip`** afterward. It contains all saved
matrices, right-hand sides, iterate arrays, cases, failures, method metadata,
source snapshots, configurations, tables, figures, and any high-precision audit.
The CUDA bridge is never silently substituted by a CPU run; any bridge failure is
logged and included while completed core results are preserved.

A smaller alternative is:

```bash
bash scripts/run_quick.sh results/quick
```

Runs resume only when source and configuration hashes match. Keep completed runs
unchanged; use another output directory after editing code or configuration.

## Available suites

| Config | Purpose | Core arithmetic |
|---|---|---|
| `smoke.json` | Small known cases and precision/dtype checks | Explicit operation rounding |
| `core.json` | n=32/64/128, 7 nominal condition numbers, 3 seeds, 11 methods | Explicit operation rounding |
| `holdout.json` | Additional n=64/128, seeds 110–112, intermediate conditioning | Same fixed policy; no tuning |
| `paper_style.json` | n=100, random RHS, geometric singular values | Similar problem family, not bitwise MATLAB replication |
| `normalization.json` | Correct/incorrect handling of tiny half-solve RHS | Power-of-two normalization ablation |
| `residual_order.json` | Sequential residual model versus native BLAS residual | Labelled reduction-order sensitivity |
| `bridge.json` | Native and mixed blocked factors with FP32 working precision | Optional PyTorch CPU/CUDA; heterogeneous precision |

Default maximum is 30 correction updates (40 for paper-style). No reference or
high-precision diagnostic is used to decide the update. A computed zero residual,
unchanged FP32 iterate, or periodic cycle is recorded, not declared accurate.
See `docs/METHODS.md` for full stopping and evaluation semantics.

## Method names

`h=half`, `s=single`, `d=double`, ordered `(factorization, working, residual)`.
For the classical variants, correction triangular solves use factorization
precision too.

| Name | Factor / correction arithmetic | Working | Residual |
|---|---|---|---|
| `hss` | FP16 | FP32 | FP32 |
| **`hsd`** | **FP16** | **FP32** | **FP64** |
| `sss` | FP32 | FP32 | FP32 |
| `ssd` | FP32 | FP32 | FP64 |
| `hdd` | FP16 | FP64 | FP64 |
| `sdd` | FP32 | FP64 | FP64 |
| `direct16/32/64` | Direct explicitly rounded solve, no refinement | Documented in record | Not used |
| `input16_ssd` | FP16 input rounding only, then FP32 LU/solves | FP32 | FP64 |
| `half_factors_solve32` | Actual FP16 factors, applied with FP32 triangular solves | FP32 | FP64 |

The last two are **ablations**, not alternative spellings of `hsd`.
`hsd_no_norm` removes correction-RHS normalization. `*_blas` changes only residual
reduction order and retains the stated dtype.

## Reading results

Start with `report/READOUT.md`, `method_summary.csv`, `paired_comparisons.csv`, and
`histories.csv`. A predetermined teaching trace is provided in `teaching_trace.csv`.
It is selected by seed=0, nominal kappa=1000, n closest to 64—not by hunting for the
best result. The precision-to-precision comparison uses exactly the same stored
FP32 A,b. FP64-working variants promote these values exactly, isolating working
arithmetic from input-data changes.

`10u` and `100u` are predeclared descriptive forward-error thresholds, **not theorem
constants**. The target for `hsd` is working/single accuracy. A universal 1e-14
acceptance target would be inappropriate. The reference always solves the stored
system, not the pre-rounding manufactured system.

## Independent high-precision audit

```bash
# Add mpmath only if it is absent:
python -m pip install mpmath
python -m tpir audit results/your_machine/core --dps 80 --max-cases 8
python -m tpir report results/your_machine/core
```

The selected cases are deterministic and independent of observed success. The
saved output vectors are audited; the solver is not re-run in high precision.
The full archive includes the decimal reference solutions and residual metrics.
A missing mpmath installation is recorded as a skipped audit, not silently replaced.

## Optional GPU bridge only

The core pure-half tests do not need a GPU. To add only the practical implementation
comparison using the environment that already ran the earlier V100 experiments:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m tpir bridge --device cuda --out results/cuda_bridge --resume
python -m tpir report results/cuda_bridge
python -m tpir collect results/cuda_bridge --out results/cuda_bridge_results.zip
```

Bridge outputs are small accuracy experiments, **not speedup measurements**. They
include diagnostic transfers. Tensor Core execution is not inferred from dtype or
from a trace of a different run. The core file `tpir/arithmetic.py` never imports
PyTorch, so CUDA/driver compatibility cannot block the main numerical tests.

## Presentation update

Read `PRESENTATION_UPDATE_PROMPT.md` and `LOCAL_RESULTS.md` before modifying the
presentation. Make the three roles and the `hss` vs `hsd` comparison central. Retain
blocked GPU results as a distinct implementation extension. Do not mix software
arithmetic curves into the previous hardware timing plots.

## Attribution and scope

Algorithm: Carson & Higham (2018), Algorithm 1.1 and Section 6; Higham (2019),
Algorithm 2.1 and its tables. These are new implementations and experiments,
not a byte-for-byte replication of the paper. This addition does **not** implement
the paper's GMRES-IR algorithm. The prior project's FGMRES experiments remain
separate. See `docs/SOURCES.md` and `docs/METHODS.md`.

The source and tests were generated with AI assistance and should be reviewed,
understood and acknowledged in the project. Vendored factor/GEMM code comes from
the earlier project package; its license is retained.
