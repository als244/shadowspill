#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run in the allocated codex GPU pane}"
export PATH=/home/as1669/.conda/envs/shadowspill/bin:$PATH
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD SHADOWSPILL_LIBRARY_DIRECTORY
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export TMPDIR="/tmp/qwen_ep8_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd /home/as1669/shadowspill
python -u docs/internal/plans/qwen_moe_ep8_1007/scripts/sweep.py "$@"
