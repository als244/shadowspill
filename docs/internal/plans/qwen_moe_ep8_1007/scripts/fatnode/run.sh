#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/shadowspill"
plan=docs/internal/plans/qwen_moe_ep8_1007
mkdir -p "$plan/logs/fatnode"
for mode in health auto recompute; do
    printf '[%s] START %s\n' "$(date --iso-8601=seconds)" "$mode"
    "$HOME/miniconda3/envs/shadowspill/bin/python" -u "$plan/scripts/fatnode/container.py" "$mode" 2>&1 | tee "$plan/logs/fatnode/$mode.log"
    printf '[%s] PASS %s\n' "$(date --iso-8601=seconds)" "$mode"
done
