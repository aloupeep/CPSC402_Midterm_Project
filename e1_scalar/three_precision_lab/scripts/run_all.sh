#!/usr/bin/env bash
# Keep your existing NumPy/SciPy environment. No package installation/upgrade.
# Usage: bash scripts/run_all.sh results/my_machine [--cuda]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/my_machine}"
MODE="${2:-}"
if [[ -n "$MODE" && "$MODE" != "--cuda" ]]; then echo 'Second argument must be --cuda or omitted' >&2; exit 2; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$OUT"
python -m tpir doctor --out "$OUT/doctor.json" | tee "$OUT/doctor.log"
python -m unittest discover -s tests -v 2>&1 | tee "$OUT/tests.log"
python -m pip freeze > "$OUT/environment.txt"
# Every suite is resumable only with identical source and configuration hashes.
for suite in smoke core holdout normalization residual_order paper_style; do
  python -m tpir run --config "configs/$suite.json" --out "$OUT/$suite" --resume | tee "$OUT/${suite}.log"
  python -m tpir report "$OUT/$suite"
  python -m tpir plot "$OUT/$suite"
done
if python -c 'import mpmath' >/dev/null 2>&1; then
  python -m tpir audit "$OUT/core" --dps 80 --max-cases 8 | tee "$OUT/audit.log"
  python -m tpir report "$OUT/core"
else
  echo 'mpmath not installed: independent high-precision audit SKIPPED (not silently substituted).' | tee "$OUT/AUDIT_SKIPPED.txt"
fi
if [[ "$MODE" == '--cuda' ]]; then
  # An explicit hardware failure must not silently become a CPU bridge result.
  if python -m tpir bridge --device cuda --out "$OUT/cuda_bridge" --resume 2>&1 | tee "$OUT/cuda_bridge.log"; then
    python -m tpir report "$OUT/cuda_bridge"
  else
    echo 'CUDA bridge FAILED. CPU numerical experiments remain available; inspect cuda_bridge.log.' | tee "$OUT/CUDA_BRIDGE_FAILED.txt"
  fi
fi
python -m tpir collect "$OUT" --out "${OUT%/}_results.zip"
printf '\nUPLOAD: %s\n' "${OUT%/}_results.zip"
