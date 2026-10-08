#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/grad_school/research/shadowspill"
root="$PWD/docs/internal/plans/lora_models_1008"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. TORCHINDUCTOR_COMPILE_THREADS=4
trap 'result=$?; echo "$result" > "$root/logs/timing-followup.exit"' EXIT
for mode in lora lora_head; do
for variant in save recompute; do
  outdir="$root/evidence/llama-1b-timing/$mode-$variant"
  if [ -f "$outdir/result.json" ]; then
    continue
  fi
  mkdir -p "$outdir"
  "$HOME/miniconda3/envs/shadowspill/bin/python" -u "$root/scripts/full_model_lora.py" \
    --preset 1b --family llama3 --mode "$mode" --variant "$variant" --dtype bfloat16 \
    --tokens 2048 --sequence-length 512 --steps 40 --warmup 10 --warmup-seconds 2 \
    --record-gc --execution-gib 16 --outdir "$outdir" 2>&1 | tee "$outdir/console.log"
done
done
"$HOME/miniconda3/envs/shadowspill/bin/python" -u "$root/scripts/probe_dense_epilogue.py" \
  2>&1 | tee "$root/logs/dense-epilogue.log"
