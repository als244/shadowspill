#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/grad_school/research/shadowspill"
root="$PWD/docs/internal/plans/lora_models_1008"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
trap 'result=$?; echo "$result" > "$root/logs/full-lora-cpu.exit"' EXIT
"$HOME/miniconda3/envs/shadowspill/bin/python" -m pytest -o addopts= -x -q \
  --junitxml="$root/evidence/full-lora-cpu.xml" tests/workloads/test_lora_models.py \
  2>&1 | tee "$root/logs/full-lora-cpu.log"
