#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/grad_school/research/shadowspill"
root="$PWD/docs/internal/plans/lora_models_1008"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. TORCHINDUCTOR_COMPILE_THREADS=4
trap 'result=$?; echo "$result" > "$root/logs/full-smoke.exit"' EXIT
"$HOME/miniconda3/envs/shadowspill/bin/python" -u "$root/scripts/run_full_lora_sweep.py" --suite smoke --resume \
  2>&1 | tee "$root/logs/full-smoke.log"
