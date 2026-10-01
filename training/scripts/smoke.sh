#!/usr/bin/env bash
# Smoke test of the whole pipeline on both backends: six steps with an
# evaluation and a checkpoint every three, a resume to eight, then
# `training.compare` (the same documents at every step? the same losses?).
#   training/scripts/smoke.sh <run dir prefix> <config.json>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$(dirname "$1")"
prefix="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
config=$2
for backend in pytorch shadowspill; do
    if [ -e "${prefix}_$backend" ]; then
        echo "${prefix}_$backend exists; a smoke test starts fresh" >&2
        exit 1
    fi
done
short=(eval_every=3 eval_batches=2 checkpoint_every=3 wandb_project=null)
pytorch='backend={"@call": "shadowspill.training.backends:PyTorch"}'
for backend in pytorch shadowspill; do
    extra=()
    if [ "$backend" = pytorch ]; then extra+=("$pytorch"); fi
    run_dir="${prefix}_$backend"
    "$here/launch.sh" "$run_dir" "$config" "${extra[@]}" steps=6 "${short[@]}"
    "$here/launch.sh" "$run_dir" "$config" "${extra[@]}" steps=8 "${short[@]}" \
        "resume=$run_dir/checkpoints/step_00000006"
done

cd "$here/../.."
"${PYTHON:-python}" -m training.compare "${prefix}_pytorch" "${prefix}_shadowspill"
