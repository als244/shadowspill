#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside codex GPU allocation}"
export PATH=/home/as1669/.conda/envs/shadowspill/bin:$PATH
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD SHADOWSPILL_LIBRARY_DIRECTORY TMPDIR
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
cd /home/as1669/shadowspill
out=/home/as1669/storage/shadowspill/generic_training_0930/della_1001/saved-value-lifetimes
mkdir -p "$out"
for case in \
  test_a_plan_keeps_no_host_copy_of_what_its_forwards_saved \
  test_planning_fails_when_the_spill_pool_has_no_room_for_saved_values; do
  python -u -m pytest -o addopts='' -q -s "tests/shadowspill/pytorch/api/test_03_planning_host_memory.py::$case" 2>&1 | tee "$out/$case.log"
done
