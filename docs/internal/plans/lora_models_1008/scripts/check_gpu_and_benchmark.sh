#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
root=/home/shein/Documents/grad_school/research/shadowspill/docs/internal/plans/lora_models_1008
cd /home/shein/Documents/grad_school/research/mlops
/home/shein/miniconda3/envs/shadowspill/bin/python -m pytest -q tests/test_lora_modules.py tests/test_lora_head.py tests/test_custom_op_contracts.py -k 'lora or head or manifest or schema_is_functional' \
  2>&1 | tee "$root/logs/lora-gpu-regression.log"
bash "$root/scripts/run_full_benchmark.sh"
