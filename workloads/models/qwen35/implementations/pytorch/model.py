"""Pure-PyTorch Qwen 3.5 blocks and decoder assembly."""

from __future__ import annotations

import torch
from torch import nn

from workloads.models.qwen35.lora import SHARED_TARGETS as LORA_SHARED_TARGETS
from workloads.models.qwen35.lora import TARGETS as LORA_TARGETS
from workloads.models.shared.decoder import (
    RMSNorm,
    RotaryEmbedding,
    SequenceLengths,
    attention_metadata,
    language_model_loss,
    swiglu,
)

from ...config import Config
from .attention import GatedAttention
from .linear_attention import GatedDeltaNet


class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(hidden), self.w3(hidden)))


class Block(nn.Module):
    def __init__(self, config: Config, index: int) -> None:
        super().__init__()
        self.kind = config.layer_kind(index)
        self.attn_norm = RMSNorm(
            config.d_model,
            epsilon=config.norm_epsilon,
            zero_centered=config.zero_centered_norm,
        )
        self.mixer: nn.Module = (
            GatedAttention(config) if self.kind == "full" else GatedDeltaNet(config)
        )
        self.ffn_norm = RMSNorm(
            config.d_model,
            epsilon=config.norm_epsilon,
            zero_centered=config.zero_centered_norm,
        )
        self.mlp = MLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        lengths: tuple[int, ...],
        rotary: RotaryEmbedding,
    ) -> torch.Tensor:
        normalized = self.attn_norm(hidden)
        if self.kind == "full":
            update = self.mixer(normalized, positions, lengths, rotary)
        else:
            update = self.mixer(normalized, lengths)
        hidden = hidden + update
        return hidden + self.mlp(self.ffn_norm(hidden))


class Model(nn.Module):
    SUPPORTS_PACKED = True

    LORA_TARGETS = LORA_TARGETS
    LORA_SHARED_TARGETS = LORA_SHARED_TARGETS

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(
            config.rotary_width, base=config.rope_base, capacity=config.max_seq_len
        )
        self.blocks = nn.ModuleList(
            Block(config, index) for index in range(config.n_layers)
        )
        self.final_norm = RMSNorm(
            config.d_model,
            epsilon=config.norm_epsilon,
            zero_centered=config.zero_centered_norm,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tied_embeddings:
            self.lm_head.weight = self.embed.weight

    def hidden(
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> torch.Tensor:
        positions, lengths = attention_metadata(
            tokens, sequence_lengths, capacity=self.config.max_seq_len
        )
        hidden = self.embed(tokens)
        for block in self.blocks:
            hidden = block(hidden, positions, lengths, self.rotary)
        return self.final_norm(hidden)

    def forward(
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> torch.Tensor:
        return self.lm_head(self.hidden(tokens, sequence_lengths))

    def loss(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        *,
        seq_lens: SequenceLengths = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        return language_model_loss(
            self.hidden(tokens, seq_lens), self.lm_head, targets, reduction
        )
