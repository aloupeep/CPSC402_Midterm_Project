#!/bin/bash
# Final assembly: audit -> collect -> summarize -> audit summary -> run summary -> finalize -> docs -> validate -> package
set -e
cd /home/worker/lab/campaign
PY=/home/worker/lab/venv/bin/python
$PY audit.py "$1"
python3 collect.py
$PY summarize.py
python3 dev_table.py
python3 audit_summary.py
cp docs_static/LIMITATIONS.md docs_static/ATTRIBUTION.md docs_static/literature_mapping.md writeup_results/
python3 write_run_summary.py
python3 finalize.py
rm -f /home/worker/lab/mixed_precision_writeup_results.zip
python3 /home/worker/lab/handoff/mixed_precision_writeup_handoff/tools/validate_results.py writeup_results
python3 /home/worker/lab/handoff/mixed_precision_writeup_handoff/tools/package_results.py writeup_results --out /home/worker/lab/mixed_precision_writeup_results.zip
