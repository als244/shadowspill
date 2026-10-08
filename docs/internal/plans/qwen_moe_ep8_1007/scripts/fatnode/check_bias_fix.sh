#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/shadowspill"
plan=docs/internal/plans/qwen_moe_ep8_1007
python3 "$plan/scripts/fatnode/container.py" bias --world-size 4 --probe-args --variant recompute 2>&1 | tee "$plan/logs/fatnode/dp4/bias-recompute-fixed.log"
python3 "$plan/scripts/fatnode/container.py" bias --world-size 4 --probe-args --no-symmetric-planning 2>&1 | tee "$plan/logs/fatnode/dp4/bias-independent-fixed.log"
python3 "$plan/scripts/fatnode/container.py" suite --world-size 2 2>&1 | tee "$plan/logs/fatnode/suite-functionalized.log"
