#!/usr/bin/env bash
set -euo pipefail
cd /home/shein/Documents/grad_school/research/shadowspill
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
root="$PWD/docs/internal/plans/lora_models_1008"
/home/shein/miniconda3/envs/shadowspill/bin/python -u "$root/scripts/diagnose_alias.py" --family qwen3moe --rank 4 --mode lora_head --outdir "$root/evidence/full-model-smoke/mlops-qwen3moe-lora_head-save" 2>&1 | tee "$root/logs/qwen-alias-diagnosis.log"
