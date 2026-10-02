#!/usr/bin/env bash
set -uo pipefail
: "${SLURM_JOB_ID:?Run in the allocated codex GPU pane}"
export PATH=/home/as1669/.conda/envs/shadowspill/bin:$PATH
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD SHADOWSPILL_LIBRARY_DIRECTORY
if [[ -n "${PROBE_DSL_ROOT:-}" ]]; then
  export PYTHONPATH="$PROBE_DSL_ROOT/nvidia_cutlass_dsl/python_packages:$PROBE_DSL_ROOT"
fi
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
cd /home/as1669/shadowspill
phase="$PWD/docs/internal/plans/generic_training_0930/della_1001/scripts"
base=/home/as1669/storage/shadowspill/generic_training_0930/della_1001
extra=()
extra+=(--experts "${PROBE_EXPERTS:-192}" --top-k "${PROBE_TOP_K:-4}" --dim "${PROBE_DIM:-1024}")
extra+=(--iterations "${PROBE_ITERATIONS:-1}")
if [[ "${PROBE_CHECK_ROUNDTRIP:-0}" == 1 ]]; then
  extra+=(--check-roundtrip)
fi
if [[ "${PROBE_EXACT_PAYLOAD:-0}" == 1 ]]; then
  extra+=(--exact-payload)
fi
if [[ -n "${PROBE_COMPILER_OPTIONS:-}" ]]; then
  extra+=("--compiler-options=$PROBE_COMPILER_OPTIONS")
fi
if [[ "${PROBE_MEMORY_CLOBBER:-0}" == 1 ]]; then
  extra+=(--memory-clobber)
fi
failed=0
for tokens in ${PROBE_TOKEN_CASES:-32768 65536}; do
  out="$base/stock-moonep-${SLURM_JOB_ID}-${PROBE_LABEL:-current}-t${tokens}"
  mkdir -p "$out"
  timeout --signal=TERM --kill-after=5 120 \
    python -u -m torch.distributed.run --standalone --nproc-per-node=2 --tee 3 \
    --log-dir "$out/processes" "$phase/repro_moonep_64k.py" \
    --tokens "$tokens" --outdir "$out" "${extra[@]}" 2>&1 | tee "$out/console.log" || failed=1
done
exit "$failed"
