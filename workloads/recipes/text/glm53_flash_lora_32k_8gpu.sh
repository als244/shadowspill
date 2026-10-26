#!/usr/bin/env bash
# Run from the ShadowSpill checkout in its installed training environment.
set -euo pipefail
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
exec python -m torch.distributed.run --standalone --nproc-per-node=8 \
  --module benchmarking.device_worker \
  workloads.recipes.text.train_glm53_flash_lora \
  --sequence-length 32768 --sequences-per-step 1 \
  --max-sequences-per-microbatch 1 \
  --execution-gib 76 --spill-gib 96 \
  --steps 100 --schedule-steps 100 --eval-every 10 --checkpoint-every 10 \
  "$@"
