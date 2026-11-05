# Primary sources and what they support

1. Erin Carson and Nicholas J. Higham (2018). *Accelerating the Solution of Linear
   Systems by Iterative Refinement in Three Precisions*. SIAM Journal on Scientific
   Computing 40(2), A817–A847. DOI 10.1137/17M1140819.
   https://eprints.maths.manchester.ac.uk/2629/1/cahi18.pdf
   Algorithm 1.1: precision roles. Section 6: residual normalization and half range.
   Sections 7 and 10: classical LU refinement and its experiments. Section 8 is
   GMRES-IR; that distinct algorithm is NOT newly implemented in this package.

2. Nicholas J. Higham (2019). *Error Analysis For Standard and GMRES-Based Iterative
   Refinement in Two and Three-Precisions*. MIMS EPrint 2019.19.
   https://eprints.maths.manchester.ac.uk/2735/1/paper.pdf
   Algorithm 2.1 and Tables 2.1/2.2 summarize standard refinement. The tabulated
   condition-number scales omit dimension factors and other terms. Half/single/
   double aims for single-precision-level accuracy under the stated assumptions.

3. Prior project implementation `mixed_precision_lab_v1_1`, source supplied in this
   conversation. `tpir/vendor/factors.py` and `tpir/vendor/gemm.py` are copied from
   it, with their license retained in `VENDORED_LICENSE`. They are used only by
   the optional heterogeneous blocked bridge, not the pure-half arithmetic model.

Our numerical data, tests, arithmetic operation choices, exact-rational oracle,
reference audits, configs and plots are project artifacts, not results attributed
to either paper. The primary-source descriptions above do not promise a speedup
or imply our generated matrices reproduce the paper's exact figures.
