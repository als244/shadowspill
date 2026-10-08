#!/usr/bin/env bash
set -euo pipefail
root="$HOME/Documents/grad_school/research/shadowspill/docs/internal/plans/lora_models_1008"
cd "$root/scripts"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$HOME/miniconda3/envs/shadowspill/bin/python" -m pytest -o addopts= -q -s --junitxml="$root/evidence/head-cpu.xml" test_head_candidate.py 2>&1 | tee "$root/logs/head-cpu-prototype.log"
