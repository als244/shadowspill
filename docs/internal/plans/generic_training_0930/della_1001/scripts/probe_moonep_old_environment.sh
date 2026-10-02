#!/usr/bin/env bash
# Historical comparison only: no production dependency on dev/moe_lab.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside the allocated codex pane}"
source /home/as1669/shadowspill/dev/moe_lab/scripts/environment.sh quack
out="/home/as1669/storage/shadowspill/generic_training_0930/della_1001/stock-moonep-${SLURM_JOB_ID}-${PROBE_LABEL:-old-moe-lab}-t65536"
mkdir -p "$out"
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
timeout --signal=TERM --kill-after=5 120 \
  python -u -m torch.distributed.run --standalone --nproc-per-node=2 --tee 3 \
  --log-dir "$out/processes" \
  /home/as1669/shadowspill/docs/internal/plans/generic_training_0930/della_1001/scripts/repro_moonep_64k.py \
  --tokens 65536 --outdir "$out" 2>&1 | tee "$out/console.log"
