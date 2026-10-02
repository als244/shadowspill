#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside codex GPU allocation}"
export PATH=/home/as1669/.conda/envs/shadowspill/bin:$PATH
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD SHADOWSPILL_LIBRARY_DIRECTORY
export OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
cd /home/as1669/shadowspill
store=/home/as1669/storage/shadowspill/generic_training_0930/della_1001
out="$store/chicago-ep2-capacity-fixes-1002"
export TMPDIR="/tmp/ss_capacity_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$store/chicago-olmoe12b-ep2-1002/inductor_cache"
export TRITON_CACHE_DIR="$store/chicago-olmoe12b-ep2-1002/triton_cache"
export CUDA_CACHE_PATH="$store/chicago-olmoe12b-ep2-1002/cuda_cache"
mkdir -p "$out" "$TMPDIR"
git rev-parse HEAD > "$out/shadowspill-revision.txt"
git -C /home/as1669/mlops rev-parse HEAD > "$out/mlops-revision.txt"
git diff > "$out/shadowspill-changes.patch"
git -C /home/as1669/mlops diff > "$out/mlops-changes.patch"
python -u docs/internal/plans/generic_training_0930/della_1001/scripts/retry_chicago_capacities.py "$@" 2>&1 | tee -a "$out/console.log"
