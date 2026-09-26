#!/usr/bin/env bash
# Train one run:  training/scripts/launch.sh <run dir> <config.json> [key=value ...]
#
# Runs `python -m training.train` from the repository root, so a config's
# relative paths are the repository's, and keeps everything the run prints --
# ShadowSpill's own output included -- in <run dir>/stdout.log as well.
# Launching an existing run directory resumes it from its checkpoint. PYTHON
# picks the interpreter (default: python).
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$1"
run_dir="$(cd "$1" && pwd)"
config="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
cd "$repo"
"${PYTHON:-python}" -u -m training.train "$config" "run_dir=$run_dir" "${@:3}" 2>&1 \
    | tee -a "$run_dir/stdout.log"
