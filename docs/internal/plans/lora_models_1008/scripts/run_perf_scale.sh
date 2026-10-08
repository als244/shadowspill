#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/grad_school/research/shadowspill"
plan="$PWD/docs/internal/plans/lora_models_1008"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. TORCHINDUCTOR_COMPILE_THREADS=4
trap 'result=$?; echo "$result" > "$plan/logs/perf-scale.exit"' EXIT
"$HOME/miniconda3/envs/shadowspill/bin/python" -u "$plan/scripts/run_perf_scale.py" --resume \
  2>&1 | tee -a "$plan/logs/perf-scale.log"
