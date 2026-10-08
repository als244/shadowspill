#!/usr/bin/env bash
set -euo pipefail
root="$HOME/Documents/grad_school/research/shadowspill/docs/internal/plans/lora_models_1008"
cd "$root/scripts"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$HOME/miniconda3/envs/shadowspill/bin/python" -m pytest -q -s test_lora_prototype.py 2>&1 | tee "$root/logs/cpu-prototype.log"
