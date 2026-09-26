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
pytorch='backend={"@call": "training.backends.pytorch:PyTorch"}'
for steps in 6 8; do  # the second pass resumes from the step-6 checkpoint
    "$here/launch.sh" "${prefix}_pytorch" "$config" "$pytorch" "steps=$steps" "${short[@]}"
    "$here/launch.sh" "${prefix}_shadowspill" "$config" "steps=$steps" "${short[@]}"
done
cd "$here/../.."
"${PYTHON:-python}" -m training.compare "${prefix}_pytorch" "${prefix}_shadowspill"
