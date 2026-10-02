#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run in codex allocation}"
export PATH=/home/as1669/.conda/envs/shadowspill/bin:$PATH
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD SHADOWSPILL_LIBRARY_DIRECTORY TMPDIR
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
out=/home/as1669/storage/shadowspill/generic_training_0930/della_1001/moonep-planner-o2-validation
mkdir -p "$out"
cd /home/as1669/mlops
if [[ ${1:-} != --shadowspill-only ]]; then
  python -u -m pytest -q -s tests/expert_parallel/test_gpu.py --run-expert-parallel --ep-backend both --ep-world-size 2 --ep-output "$out/mlops" 2>&1 | tee "$out/mlops.log"
fi
cd /home/as1669/shadowspill
phase="$PWD/docs/internal/plans/generic_training_0930/della_1001"
for variant in save recompute; do
  python -u -m torch.distributed.run --standalone --nproc-per-node=2 --tee 3 --log-dir "$out/$variant/processes" "$phase/scripts/run_quack_ep.py" --variant "$variant" --outdir "$out/$variant" --router-dtype bfloat16 --share-buffer 2>&1 | tee "$out/$variant.log"
  python -u "$phase/scripts/compare_quack_ep.py" "$out/$variant" 2>&1 | tee "$out/$variant/oracle.log"
done
python -c "from pathlib import Path; Path('$out/PASSED').touch()"
