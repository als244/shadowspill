#!/usr/bin/env bash
set -euo pipefail
root="$HOME/Documents/grad_school/research/shadowspill/docs/internal/plans/lora_models_1008"
cd "$root/staged_mlops"
export PYTHONPATH="$root/staged_mlops/src"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$HOME/miniconda3/envs/shadowspill/bin/python" -m pytest -o addopts= -q -s -m "not gpu" --junitxml="$root/evidence/lora-head-cpu.xml" tests/test_lora_head.py 2>&1 | tee "$root/logs/lora-head-cpu.log"
