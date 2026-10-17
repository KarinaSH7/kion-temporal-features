#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m src.data --users 10000
"$PYTHON" -m pytest -q
"$PYTHON" -u -m src.experiment build-dev
"$PYTHON" -u -m src.audit
"$PYTHON" -u -m src.experiment select
"$PYTHON" -u -m src.experiment final-test
"$PYTHON" -u -m src.audit --include-test
"$PYTHON" -u -m src.benchmark
"$PYTHON" -m src.report
"$PYTHON" -m src.plot_results
