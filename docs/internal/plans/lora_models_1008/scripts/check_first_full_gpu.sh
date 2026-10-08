#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/grad_school/research/shadowspill"
root="$PWD/docs/internal/plans/lora_models_1008"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.
export TORCHINDUCTOR_COMPILE_THREADS=4
trap 'result=$?; echo "$result" > "$root/logs/first-full-gpu.exit"' EXIT
"$HOME/miniconda3/envs/shadowspill/bin/python" -u "$root/scripts/full_model_lora.py" \
 --family llama3 --mode lora_head --variant save --preset smoke --rank 4 \
 --outdir "$root/evidence/full-model-smoke/mlops-llama3-lora_head-save" \
 2>&1 | tee "$root/logs/first-full-gpu.log"
