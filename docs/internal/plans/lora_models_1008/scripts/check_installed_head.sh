#!/usr/bin/env bash
set -euo pipefail
root="$HOME/Documents/grad_school/research/shadowspill/docs/internal/plans/lora_models_1008"
cd "$HOME/Documents/grad_school/research/mlops"
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$HOME/miniconda3/envs/shadowspill/bin/python" -c 'import mlops; print("Installed module:", mlops.__file__)'
"$HOME/miniconda3/envs/shadowspill/bin/python" -m pytest -o addopts= -q -m 'not gpu' tests/test_lora_head.py tests/test_head.py tests/test_flop_formulas.py tests/test_documentation.py tests/test_implementation_registry.py --junitxml="$root/evidence/installed-head-cpu.xml" 2>&1 | tee "$root/logs/installed-head-cpu.log"
