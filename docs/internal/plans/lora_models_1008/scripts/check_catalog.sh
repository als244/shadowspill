#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/grad_school/research/shadowspill"
root="$PWD/docs/internal/plans/lora_models_1008"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
trap 'result=$?; echo "$result" > "$root/logs/catalog-check.exit"' EXIT
{
  "$HOME/miniconda3/envs/shadowspill/bin/python" docs/internal/plans/lora_models_1008/scripts/model_catalog.py
  "$HOME/miniconda3/envs/shadowspill/bin/python" -m pytest -o addopts= -q \
    --junitxml="$root/evidence/catalog-checks.xml" \
    tests/repository/test_documentation.py tests/workloads/test_qwen_moe.py
  "$HOME/miniconda3/envs/shadowspill/bin/python" -m benchmarking.quickstart --help
  git diff --check
} 2>&1 | tee "$root/logs/catalog-checks.log"
