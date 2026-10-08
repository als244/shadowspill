#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/grad_school/research/shadowspill"
root="$PWD/docs/internal/plans/lora_models_1008"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
trap 'result=$?; echo "$result" > "$root/logs/model-docs.exit"' EXIT
{
  "$HOME/miniconda3/envs/shadowspill/bin/python" "$root/scripts/check_model_imports.py"
  "$HOME/miniconda3/envs/shadowspill/bin/python" "$root/scripts/model_catalog.py"
  "$HOME/miniconda3/envs/shadowspill/bin/python" -m pytest -o addopts= -q \
    --junitxml="$root/evidence/model-docs-tests.xml" tests/repository/test_documentation.py
  "$HOME/miniconda3/envs/shadowspill/bin/python" -m benchmarking.quickstart --help
  "$HOME/miniconda3/envs/shadowspill/bin/ruff" check workloads/mlops tests/workloads/test_expert_parallel_construction.py tests/workloads/test_qwen_moe.py
  git diff --check
} 2>&1 | tee "$root/logs/model-docs.log"
