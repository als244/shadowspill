#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/grad_school/research/shadowspill"
root="$PWD/docs/internal/plans/lora_models_1008"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. TORCHINDUCTOR_COMPILE_THREADS=4
trap 'result=$?; echo "$result" > "$root/logs/qualification.exit"' EXIT
"$HOME/miniconda3/envs/shadowspill/bin/python" -u -m qualification.gates \
  suite numerical --run lora_models_1008 \
  2>&1 | tee "$root/logs/qualification.log"
