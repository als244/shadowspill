"""Use the existing Qwen definition and loss without a ShadowSpill dependency."""

from __future__ import annotations

import torch
from torch import nn

from workloads.mlops.qwen3_moe import Qwen3MoE, Qwen3MoEConfig


def configuration(*, tiny=False, sequence_length=4096):
    if not tiny:
        return Qwen3MoEConfig(max_seq_len=max(40960, sequence_length))
    return Qwen3MoEConfig(
        n_layers=2,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        n_experts=8,
        top_k=2,
        d_ff_expert=128,
        vocab_size=1024,
        max_seq_len=max(512, sequence_length),
    )


class TrainingModel(nn.Module):
    def __init__(
        self,
        config,
        *,
        local_tokens_per_step,
        sequence_length=4096,
        head_scratch_bytes=512 << 20,
    ):
        super().__init__()
        self.model = Qwen3MoE(config, dtype=torch.float32)
        self.local_tokens_per_step = local_tokens_per_step
        self.sequence_length = sequence_length
        rows = head_scratch_bytes // (2 * config.vocab_size)
        self.head_chunk_size = max(512, rows // 256 * 256)

    def forward(self, tokens, targets):
        return (
            self.model.loss(
                tokens,
                targets,
                seq_lens=(self.sequence_length,)
                * (tokens.numel() // self.sequence_length),
                reduction="sum",
                head_chunk_size=self.head_chunk_size,
            )
            / self.local_tokens_per_step
        )


def initialize(module):
    """Initialize the materialized shards and recompute nonpersistent buffers."""
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()
