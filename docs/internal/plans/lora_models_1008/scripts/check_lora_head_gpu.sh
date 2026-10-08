#!/usr/bin/env bash
set -euo pipefail
root="$HOME/Documents/grad_school/research/shadowspill/docs/internal/plans/lora_models_1008"
repo="$HOME/Documents/grad_school/research/shadowspill"
export PYTHONPATH="$root/staged_mlops/src:$repo"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH="/usr/local/cuda-13.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}"
export TORCHINDUCTOR_CACHE_DIR="$root/evidence/head-inductor-cache"
export TRITON_CACHE_DIR="$root/evidence/head-triton-cache"
python_bin="$HOME/miniconda3/envs/shadowspill/bin/python"
cd "$root/staged_mlops"
"$python_bin" -m pytest -o addopts= -q -m gpu tests/test_lora_head.py --junitxml="$root/evidence/lora-head-gpu.xml" 2>&1 | tee "$root/logs/lora-head-gpu.log"
"$python_bin" -m pytest -o addopts= -q tests/test_custom_op_contracts.py -k 'head or manifest' --junitxml="$root/evidence/lora-head-contracts-gpu.xml" 2>&1 | tee "$root/logs/lora-head-contracts-gpu.log"
cd "$repo"
for variant in save recompute; do
    "$python_bin" -u "$root/scripts/check_lora_head_shadowspill.py" --variant "$variant" --outdir "$root/evidence/head-shadowspill-$variant" 2>&1 | tee "$root/logs/head-shadowspill-$variant.log"
done
printf 'LoRA head GPU and ShadowSpill validation passed\n'
