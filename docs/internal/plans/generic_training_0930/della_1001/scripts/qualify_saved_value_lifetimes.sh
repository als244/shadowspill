#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside codex GPU allocation}"
export PATH=/home/as1669/.conda/envs/shadowspill/bin:$PATH
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD SHADOWSPILL_LIBRARY_DIRECTORY TMPDIR
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
cd /home/as1669/shadowspill
if (( $# == 0 )); then
  set -- suite numerical
fi
python -u -m qualification.gates "$@" --config qualification/gates_h100.json --run della_saved_lifetimes_h100_1002
