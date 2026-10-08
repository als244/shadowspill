#!/usr/bin/env bash
# Approved commit groups for 2026-10-08. Run MLOps before ShadowSpill.
set -euo pipefail
cd /home/shein/Documents/grad_school/research/mlops
git diff --check
git add -- src/mlops/kernels/moe_router.py tests/test_explicit_ops.py
git commit -m 'Keep MoE auxiliary routing counts in FP32 for every default dtype'
git add -- src/mlops/kernels/moe_grouped_gemm.py
git commit -m 'Respect precision and device limits in grouped expert GEMMs'
git add -- docs/API_REFERENCE.md docs/OPS.md docs/LORA.md src/mlops/__init__.py src/mlops/modules.py src/mlops/lora src/mlops/lora_head.py src/mlops/dispatch/logical_costs.py src/mlops/explicit/__init__.py src/mlops/explicit/lora_head_loss.py src/mlops/kernels/matmul.py src/mlops/providers/__init__.py src/mlops/providers/builtin/head.py src/mlops/providers/builtin/lora_head.py src/mlops/providers/native_torch/lora_head.py tests/test_custom_op_contracts.py tests/test_flop_formulas.py tests/test_lora_head.py tests/test_lora_modules.py
git commit -m 'Add configurable LoRA modules and omit frozen head and expert gradients'
