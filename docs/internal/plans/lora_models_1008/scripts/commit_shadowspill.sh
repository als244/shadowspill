#!/usr/bin/env bash
# Approved commit groups for 2026-10-08. Requires the new MLOps LoRA modules.
set -euo pipefail
cd /home/shein/Documents/grad_school/research/shadowspill
git diff --check
git add -- src/shadowspill/pytorch/capture/aot.py src/shadowspill/pytorch/capture/materialize.py src/shadowspill/pytorch/graph_pairs/artifacts.py tests/shadowspill/pytorch/graph_pairs/test_graph_pairs.py
git commit -m 'Materialize shared activation cotangents at task boundaries'
git add -- src/shadowspill/pytorch/profiling/profiler/workspace.py src/shadowspill/task/layout.py tests/shadowspill/pytorch/compilation/test_layout.py tests/shadowspill/pytorch/profiling/test_profiler.py
git commit -m 'Account for empty tensor views of nonempty allocations'
git add -- workloads/README.md workloads/MODELS.md workloads/full_model.py workloads/mlops/__init__.py workloads/mlops/llama3.py workloads/mlops/olmoe.py workloads/mlops/qwen35.py workloads/mlops/qwen_moe workloads/mlops/_expert_parallel.py workloads/mlops/_qwen_moe workloads/mlops/qwen3_moe.py workloads/mlops/qwen35_moe.py workloads/quack workloads/lora workloads/recipes/text/models.py tests/workloads/test_qwen_moe.py tests/workloads/test_expert_parallel_construction.py tests/workloads/test_lora_models.py benchmarking/quickstart/options.py README.md docs/README.md docs/examples/README.md training/README.md
git add -f -- docs/internal/plans/generic_training_0930/della_1001/scripts/chicago_ep2/train.py docs/internal/plans/generic_training_0930/della_1001/scripts/run_quack_ep.py docs/internal/plans/qwen_moe_ep8_1007/scripts/experiment.py docs/internal/plans/qwen_moe_ep8_1007/scripts/prepare_data.py docs/internal/plans/qwen_moe_ep8_1007/scripts/sweep.py docs/internal/plans/qwen_moe_ep8_1007/scripts/train.py
git commit -m 'Organize example architectures and expose full-model LoRA recipes'
# Evidence, raw artifact stores and staged source copies are deliberately not
# swept into git. Review RESULTS.md and the small report tables separately.
