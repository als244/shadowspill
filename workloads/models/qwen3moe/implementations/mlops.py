"""Qwen 3 MoE on mlops operations: GQA and routed experts in every block,
with optional expert parallelism."""

from __future__ import annotations

from workloads.models.qwen3moe.lora import SHARED_TARGETS as LORA_SHARED_TARGETS
from workloads.models.qwen3moe.lora import TARGETS as LORA_TARGETS
from workloads.models.shared.qwen_moe.model import QwenMoE

from ..config import Config


class Model(QwenMoE):
    LORA_TARGETS = LORA_TARGETS
    LORA_SHARED_TARGETS = LORA_SHARED_TARGETS

    def __init__(self, config=None, **kwargs):
        super().__init__(config or Config(), **kwargs)


__all__ = ["Model"]
