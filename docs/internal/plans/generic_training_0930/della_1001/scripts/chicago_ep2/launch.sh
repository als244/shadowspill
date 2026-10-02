#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_ID:-}" == 14860963 ]] || { echo 'Expected Della allocation 14860963'; exit 2; }
cd /home/as1669/shadowspill
recipe="$PWD/docs/internal/plans/generic_training_0930/della_1001/scripts/chicago_ep2"
out=/home/as1669/storage/shadowspill/generic_training_0930/della_1001/chicago-olmoe12b-ep2-1002
export PATH=/home/as1669/.conda/envs/shadowspill/bin:$PATH
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD SHADOWSPILL_LIBRARY_DIRECTORY
export OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export HTTP_PROXY=http://127.0.0.1:18375 HTTPS_PROXY=http://127.0.0.1:18375
export NO_PROXY=localhost,127.0.0.1
export WANDB_DIR="$out" WANDB_MODE=online
export TMPDIR="$out/tmp" TORCHINDUCTOR_CACHE_DIR="$out/inductor_cache"
export TRITON_CACHE_DIR="$out/triton_cache" CUDA_CACHE_PATH="$out/cuda_cache"
mkdir -p "$out" "$TMPDIR"
[[ ! -e "$out/exit.json" ]] || { echo 'Use a fresh output directory'; exit 2; }
trap 'code=$?; python - "$out/exit.json" "$code" <<"PY"
from pathlib import Path
from datetime import datetime, timezone
import json,sys
Path(sys.argv[1]).write_text(json.dumps({"exit_code":int(sys.argv[2]),"passed":int(sys.argv[2])==0,"utc":datetime.now(timezone.utc).isoformat()})+"\n")
PY
' EXIT
git rev-parse HEAD > "$out/shadowspill_revision.txt"
git -C /home/as1669/mlops rev-parse HEAD > "$out/mlops_revision.txt"
cp "$recipe/config.json" "$recipe/train.py" "$out/"
python -u -m torch.distributed.run --standalone --nproc-per-node=2 \
  --tee 3 --log-dir "$out/processes" "$recipe/train.py" \
  --config "$recipe/config.json" "$@" 2>&1 | tee "$out/console.log"
