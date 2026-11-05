# writeup_results — mixed-precision LU and iterative refinement campaign (2026-11-04)

Start with `RUN_SUMMARY.md` (what ran, largest sizes, backends, qualification and timing tables), then `LIMITATIONS.md`, `methods.json` (precision contracts), `coverage.csv`, and `summaries/` (every table behind the figures; `claims.json` is the machine-readable claim ledger).

Layout follows `docs/RESULTS_CONTRACT.md` of the handoff:
- `cases.jsonl` — one stored problem per line (hashes, generator recipe, conditioning estimates, reference metadata, input evidence).
- `trials.jsonl` — every run of every method (timed, warmup, diagnostic, reuse, profile), with original-system metrics, validation, counters, timings, saved solution path. `suite` ∈ {dev, E2, E2a, E3, E4, E5, E6, E8, E1}.
- `histories.jsonl` — outer/inner iteration records keyed by `trial_id`.
- `solutions/` — every final X (deduplicated by SHA-256; raw little-endian column-major float32/float64 for the GPU track, `.npy` for E1).
- `rhs/`, `matrices/` (embedded A for n≤512 and E1; recipes for all), `references/` (numerical and exact reference solutions).
- `env/`, `build/`, `source/`, `configs/`, `tests/` (probes, smoke tests, audits, E0 replay), `logs/`, `profiles/`, `summaries/`, `figures/`.

Raw array formats: GPU-track `.f64`/`.f32` files are plain little-endian arrays of n×nrhs values in column-major order (nrhs from `cases.jsonl`); E1 `.npy` files load with numpy.
