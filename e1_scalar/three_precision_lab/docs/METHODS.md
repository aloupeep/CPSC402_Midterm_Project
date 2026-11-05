# Arithmetic and experiment specification

## 1. Scope and relation to the sources

The central source is Carson & Higham (2018), Algorithm 1.1. For classical LU
refinement, effective correction-solve precision is the factorization precision.
Higham (2019), Algorithm 2.1 gives the same precision-role layout. We compare the
six combinations hss, hsd, sss, ssd, hdd, sdd. Source terminology is retained:
`u_f`, `u`, `u_r` are unit roundoffs, while tuples in labels name the formats.
Unit roundoffs are 2^-11, 2^-24, 2^-53, not machine epsilon at 1.

This is a controlled numerical investigation. Differences from the paper's
experiments are explicit: NumPy generators rather than MATLAB gallery/rng,
rank-one non-fused GEPP, power-of-two RHS scaling rather than exactly the infinity
norm, chosen finite iteration limits, and our own reference/measurement pipeline.
The main grid uses a manufactured RHS rounded into the stored system; the separate
n=100 `paper_style` grid uses random RHS and geometric singular values, similar to
the source setup. No bit-for-bit reproduction or new convergence theorem is claimed.

## 2. Arithmetic model

Binary16 elementary inputs are represented exactly in binary64. Each +, -, *, /
is evaluated and immediately rounded to binary16. Partial pivoting uses all
remaining rows. Elimination does `fl(a - fl(l*u))`: separate product and subtraction,
not a fused multiply-add. Multipliers, Schur entries and factors are binary16.

The triangular solves similarly perform separately rounded multiply/subtract
updates and rounded divisions. The factors are NEVER silently promoted for the
standard `hss`/`hsd`/`hdd` solves. The named `half_factors_solve32` control deliberately
promotes those same factors to FP32 for their application.

Round-to-nearest/ties-to-even and gradual underflow are used. The test suite checks
more than 1800 random elementary-operation cases against an exact-rational nearest
binary16 oracle, and targeted ties, subnormals, overflow and non-fused cancellation.
This is an explicit arithmetic model, not a certification of all IEEE behavior or
an instruction-level emulation of a particular GPU. It does not model FTZ hardware.

FP32/64 ufunc operations use explicitly matching inputs and output dtypes. Core
residuals use sequential products and additions in the requested residual format;
there is no hidden higher-precision dot accumulator. A separate BLAS residual
ablation measures sensitivity to native reduction order. Its exact reduction tree
can vary across libraries/platforms and is not claimed to match the sequential model.

## 3. Stored problem and references

Generate A64 with prescribed singular values, then store A32. For manufactured
RHS cases form b from that A32 promoted to FP64 and a manufactured vector, then
round b to FP32. Random-RHS cases round an independent random RHS to FP32.
The exact real values of the saved A32,b32 define the target for EVERY method in
a pair, including working-FP64 controls. That is a deliberate common-data design.
It is not a rerun of the old original-FP64-input experiment.

Report both nominal kappa2 and actual kappa2/kappaInfinity of the stored matrix
(computed in FP64), since storage rounding can change conditioning. No sufficient
condition in the source is treated as a universal kappa cutoff. A quantity based on
kappa2 must not be substituted for kappaInfinity when discussing source tables.

The routine reference is FP64 LAPACK LU with up to six extended-residual refinement
updates. Extended precision is NumPy longdouble, whose actual significand size is
recorded: it is commonly 64 bits on x86 and can be only 53 bits on other systems.
This is a numerical reference, not exact ground truth. Precision-sensitive forward
accuracy claims use the independent mpmath audit. An ordinary FP64 LU result is
never called an exact solution.

The audit promotes exact stored floating values into mpmath, solves at the declared
number of decimal digits, and recomputes errors for the actual saved final iterate.
Selection is deterministic: the teaching case (seed 0, kappa 1000, n nearest 64),
then cases sorted by n, seed, nominal kappa. The default is eight cases. It does
not select only successes or suspicious outliers.

## 4. Correction normalization and initialization

For a nonzero residual with max magnitude m, choose e from frexp(m), so
0.5 <= ||2^-e r||_inf < 1. Scale in residual precision, convert to the solve precision,
apply the factors, convert to working precision, rescale with integer-exponent
ldexp in working precision, then update in working precision.

This is a power-of-two variant of the source's normalization idea. Scaling avoids
many small-RHS cast underflows; it does not guarantee safe range inside a triangular
solve or useful factors. The initial solve applies the same normalization to b.
On a failed initial triangular solve, iterative methods explicitly record a zero
initial iterate and attempt refinement; there is no hidden higher-precision
refactorization. Direct methods report their failure. A failed factorization is
not rescued. Numerical failures remain in the output.

`hsd_no_norm` removes scaling, with all other choices unchanged. It illustrates why
simply casting a tiny FP64 residual to FP16 can discard the correction information.

## 5. Stopping, evaluation, and comparisons

The solver never sees a reference solution. The main experiments aim to reveal
attainable accuracy, not optimize a threshold-based production policy. Continue
up to 30 correction updates (40 for paper_style), stopping early only for:
- a zero residual computed in the declared residual precision;
- an update that leaves the working-precision iterate exactly unchanged;
- an exact repeated iterate with period up to four;
- a detected numerical failure.

These are terminal reasons, not accuracy classifications. The returned candidate
is the LAST iterate, not the one with the smallest reference error. Histories and
first-attainment indices are saved separately. No favorable final iterate is chosen
using high-precision information.

Evaluate forward infinity-norm error, normwise backward error and componentwise
backward error on the stored problem outside the numerical solver. Normalize
errors by the method's working unit roundoff. `10u` and `100u` are declared
sensitivity/reporting levels, not universal constants from the theorems. Require
both backward measures for the combined backward flag. Do not apply the old
FP64-oriented 1e-14 criterion to an FP32 working solution.

Pairs:
- hss/hsd: same factorization hash, same correction arithmetic and working format;
  only residual precision changes.
- sss/ssd: analogous independent control with FP32 factors.
- hsd/ssd: change BOTH factorization and triangular-solve precision.
- hsd/half_factors_solve32: same half factors, different application arithmetic.
- hsd/input16_ssd: distinguishes arithmetic rounding from input-only quantization.
- hsd/hdd: change working precision on an identical stored problem; residual FP64.
- hdd/sdd: change factorization/solve precision with working/residual FP64.

Core operation order and seeds are fixed, not tuned on outcomes. The holdout grid
uses separate seeds and intermediate conditioning. It is a replication check, not
an out-of-sample performance claim for a learned/tuned method.

## 6. Practical blocked bridge

Uses vendored project v1.1 factor/GEMM code only, not its fixed-FP64 outer solver.
FP32 input data and iterates, FP32 updates, FP32 or FP64 residuals as labelled.
FP32 panel LU and block-row triangular solve; FP16 input operands and FP32
accumulation/output on selected trailing updates; FP32 subtraction and storage;
FP32 correction triangular solves. Labels state "mixed16inputs", never pure half.
CPU bridge products are a rounded-input numerical control. CUDA invokes the existing
cuBLAS wrapper. No tensor-op dispatch claim is made without a trace of that run.

No new bridge timing claim: host diagnostic copies are inside execution, dimensions
are small, and the purpose is precision-role validation. Existing V100 performance
results stay in their original experimental category and retain their limitations.

## 7. Provenance and limitations

Each run saves source/config hashes, source files, machine/package metadata, input
hashes, actual matrices, RHS, all iterates, status reasons and reference metadata.
Resume refuses changed source or configuration. Numerical failures are records;
unexpected exceptions are runtime_error records and make the campaign fail visibly.

Not covered: faithful reproduction of the paper's GMRES-IR arithmetic, IEEE
binary128 refinement, native all-half GPU LU performance, arbitrary sparse systems,
a new optimal recovery controller, or exact theorem-bound verification. These are
not silently approximated by unrelated methods.
