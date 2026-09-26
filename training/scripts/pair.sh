#!/usr/bin/env bash
# One comparison: a run on PyTorch (compiled), the same run on ShadowSpill, and
# `training.compare` over the two.
#   training/scripts/pair.sh <run dir prefix> <config.json> [key=value ...]
# writes <prefix>_pytorch, <prefix>_shadowspill and <prefix>_compare.txt.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$(dirname "$1")"
prefix="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
config=$2
shift 2
"$here/launch.sh" "${prefix}_pytorch" "$config" \
    'backend={"@call": "training.backends.pytorch:PyTorch"}' "$@"
"$here/launch.sh" "${prefix}_shadowspill" "$config" "$@"
cd "$here/../.."
"${PYTHON:-python}" -m training.compare "${prefix}_pytorch" "${prefix}_shadowspill" \
    | tee "${prefix}_compare.txt"
