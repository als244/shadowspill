"""Pure-PyTorch OLMoE decoder with functionally propagated router loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from workloads.models.olmoe.lora import SHARED_TARGETS as LORA_SHARED_TARGETS
from workloads.models.olmoe.lora import TARGETS as LORA_TARGETS
from workloads.models.shared.decoder import (
    RMSNorm,
    RotaryEmbedding,
    SequenceLengths,
    apply_rotary,
    attention_metadata,
    auxiliary_share,
    causal_attention,
    language_model_loss,
    swiglu,
)
from workloads.models.shared.routed_lora import expert_lora, factors, reset_factors

from ..config import Config


class Attention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.d_model, config.query_width, bias=False)
        self.wk = nn.Linear(config.d_model, config.key_value_width, bias=False)
        self.wv = nn.Linear(config.d_model, config.key_value_width, bias=False)
        self.q_norm = RMSNorm(config.query_width)
        self.k_norm = RMSNorm(config.key_value_width)
        self.wo = nn.Linear(config.query_width, config.d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        lengths: tuple[int, ...],
        rotary: RotaryEmbedding,
    ) -> torch.Tensor:
        config = self.config
        batch, sequence, _width = hidden.shape
        query = self.q_norm(self.wq(hidden)).view(
            batch, sequence, config.n_heads, config.head_dim
        )
        key = self.k_norm(self.wk(hidden)).view(
            batch, sequence, config.n_kv_heads, config.head_dim
        )
        value = self.wv(hidden).view_as(key)
        query = apply_rotary(query, positions, rotary)
        key = apply_rotary(key, positions, rotary)
        attended = causal_attention(query, key, value, lengths)
        return self.wo(attended.reshape(batch, sequence, config.query_width))


class MoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        gate_up, down = self.expert_shapes
        self.w13_experts = nn.Parameter(torch.empty(gate_up))
        self.w2_experts = nn.Parameter(torch.empty(down))
        self.reset_parameters()

    @property
    def expert_shapes(self):
        """The routed experts' gate/up and down banks, ``[E, out, in]``."""
        config = self.config
        return (
            (config.n_experts, 2 * config.d_ff_expert, config.d_model),
            (config.n_experts, config.d_model, config.d_ff_expert),
        )

    def reset_parameters(self) -> None:
        """Initialise the parameters this module owns, in place."""

        config = self.config
        with torch.no_grad():
            nn.init.normal_(self.w13_experts, std=config.d_model**-0.5)
            nn.init.normal_(self.w2_experts, std=config.d_ff_expert**-0.5)
        if getattr(self, "lora_config", None) is not None:
            reset_factors(self)

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        lora = getattr(self, "lora_config", None)
        if lora is not None:
            return expert_lora(
                hidden,
                residual,
                self.router.weight,
                self.w13_experts,
                self.w2_experts,
                *factors(self),
                top_k=config.top_k,
                routing_mode="softmax_then_topk",
                scale=lora.scale,
            )[:2]
        probabilities = torch.softmax(self.router(hidden).float(), dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, dim=-1, descending=True, stable=True
        )
        weights = sorted_probabilities[..., : config.top_k]
        experts = sorted_indices[..., : config.top_k]
        tokens = hidden.numel() // hidden.shape[-1]
        counts = torch.bincount(experts.reshape(-1), minlength=config.n_experts)
        frequency = counts.float() / (tokens * config.top_k)
        auxiliary = (
            config.n_experts
            * (
                frequency * probabilities.reshape(tokens, config.n_experts).mean(0)
            ).sum()
        )

        routed = torch.zeros_like(hidden, dtype=torch.float32)
        for expert in range(config.n_experts):
            coefficient = (weights * (experts == expert)).sum(dim=-1)
            gate_and_up = hidden @ self.w13_experts[expert].T
            gate, up = gate_and_up.split(  # type: ignore[no-untyped-call]
                config.d_ff_expert, dim=-1
            )
            contribution = swiglu(gate, up) @ self.w2_experts[expert].T
            routed = routed + coefficient[..., None] * contribution.float()
        return (residual.float() + routed).to(hidden.dtype), auxiliary


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.moe = MoE(config)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        lengths: tuple[int, ...],
        rotary: RotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden + self.attn(self.attn_norm(hidden), positions, lengths, rotary)
        return self.moe(self.ffn_norm(hidden), hidden)


class Model(nn.Module):
    SUPPORTS_PACKED = True

    LORA_TARGETS = LORA_TARGETS
    LORA_SHARED_TARGETS = LORA_SHARED_TARGETS

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(
            config.head_dim, base=config.rope_base, capacity=config.max_seq_len
        )
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def hidden(
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, lengths = attention_metadata(
            tokens, sequence_lengths, capacity=self.config.max_seq_len
        )
        hidden = self.embed(tokens)
        auxiliary = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for block in self.blocks:
            hidden, block_auxiliary = block(hidden, positions, lengths, self.rotary)
            auxiliary = auxiliary + block_auxiliary
        return self.final_norm(hidden), auxiliary

    def forward(
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> torch.Tensor:
        hidden, _auxiliary = self.hidden(tokens, sequence_lengths)
        return self.lm_head(hidden)

    def loss(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        *,
        seq_lens: SequenceLengths = None,
        aux_coef: float = 0.0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        hidden, auxiliary = self.hidden(tokens, seq_lens)
        objective = language_model_loss(hidden, self.lm_head, targets, reduction)
        return objective + float(aux_coef) * auxiliary_share(
            auxiliary, targets, reduction
        )


__all__ = ["Model"]
