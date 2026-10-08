#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/shadowspill"
plan=docs/internal/plans/qwen_moe_ep8_1007
mkdir -p "$plan/logs/fatnode/dp4"
for mode in health auto recompute sweep-whole sweep; do
    printf '[%s] START DP4 %s\n' "$(date --iso-8601=seconds)" "$mode"
    "$HOME/miniconda3/envs/shadowspill/bin/python" -u "$plan/scripts/fatnode/container.py" "$mode" --world-size 4 2>&1 | tee "$plan/logs/fatnode/dp4/$mode.log"
    printf '[%s] PASS DP4 %s\n' "$(date --iso-8601=seconds)" "$mode"
done
