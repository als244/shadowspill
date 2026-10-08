#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/grad_school/research/shadowspill"
root="$PWD/docs/internal/plans/lora_models_1008"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
trap 'result=$?; echo "$result" > "$root/logs/model-organization.exit"' EXIT
{
  "$HOME/miniconda3/envs/shadowspill/bin/python" "$root/scripts/local_model_regression.py" after
  "$HOME/miniconda3/envs/shadowspill/bin/python" -m pytest -o addopts= -q -m 'not cuda' \
    --junitxml="$root/evidence/model-organization-tests.xml" tests/workloads
} 2>&1 | tee "$root/logs/model-organization-tests.log"
