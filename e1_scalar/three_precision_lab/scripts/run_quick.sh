#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="${1:-results/quick}"
mkdir -p "$OUT"
python -m tpir doctor --out "$OUT/doctor.json"
python -m unittest discover -s tests -v 2>&1 | tee "$OUT/tests.log"
python -m tpir run --config configs/quick.json --out "$OUT/core" --resume
python -m tpir report "$OUT/core"
python -m tpir plot "$OUT/core"
if python -c 'import mpmath' >/dev/null 2>&1; then
 python -m tpir audit "$OUT/core" --dps 80 --max-cases 6
 python -m tpir report "$OUT/core"
fi
python -m tpir collect "$OUT" --out "${OUT%/}_results.zip"
