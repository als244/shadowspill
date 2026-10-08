"""Qwen MoE text models, independent of any planner or training backend."""

import mlops
import torch
from torch import nn

from workloads.common import RotaryEmbedding, auxiliary_share, packed_metadata

from .attention import Attention, GatedDeltaNet, RMSNorm
from .config import Qwen30BConfig, Qwen35BConfig
from .experts import MoE, parallel_options
from .initialization import Embedding, Linear


class Block(nn.Module):
    def __init__(self, config, index, **expert_options):
        super().__init__()
        self.kind = config.layer_kind(index)
        self.attn_norm = RMSNorm(config.d_model, config)
        self.mixer = Attention(config) if self.kind == "full" else GatedDeltaNet(config)
        self.ffn_norm = RMSNorm(config.d_model, config)
        self.moe = MoE(config, **expert_options)

    def forward(self, hidden, packing, rotary, cumulative, chunk_indices):
        normalized = self.attn_norm(hidden)
        if self.kind == "full":
            update = self.mixer(normalized, packing, rotary)
        else:
            update = self.mixer(normalized, cumulative, chunk_indices)
        hidden = hidden + update
        return self.moe(self.ffn_norm(hidden), hidden)


class QwenMoE(nn.Module):
    """Published text architecture with local MLOps or QuackMoE routed experts.

    For EP, supply ``ep_group`` and either ``token_capacity`` or a caller-owned
    ``buffer``. One model-owned buffer is reused across every block, including
    the expert publication banks. ``expert_parameters()`` identifies rank-unique
    parameters; all others are replicas. Communication resources are released
    by ``close()``. CPU/meta construction without EP needs no GPU dependencies.
    """

    SUPPORTS_PACKED = True

    def __init__(
        self,
        config,
        *,
        ep_group=None,
        token_capacity=None,
        buffer=None,
        device=None,
        parameter_device=None,
        dtype=None,
        compute_precision="bf16",
        weight_grad_dtype=torch.bfloat16,
        activation_transport="bf16",
    ):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()
        self._owns_buffer = False
        self._parallel = ep_group is not None
        options = None
        if self._parallel:
            if (token_capacity is None) == (buffer is None):
                raise ValueError(
                    "EP requires either token_capacity or a supplied buffer"
                )
            device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
            if device.type != "cuda":
                raise ValueError("expert parallelism requires a CUDA device")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            parameter_device = torch.device(parameter_device or device)
            dtype = dtype or torch.bfloat16
            if dtype != torch.bfloat16:
                raise ValueError("QuackMoE's hidden activation dtype must be BF16")
            options = parallel_options(
                config,
                ep_group,
                dtype=dtype,
                compute_precision=compute_precision,
                weight_grad_dtype=weight_grad_dtype,
                activation_transport=activation_transport,
            )
            if buffer is None:
                from mlops.expert_parallel import create_buffer

                buffer = create_buffer(options, token_capacity, ep_group)
                self._owns_buffer = True
        elif buffer is not None or token_capacity is not None:
            raise ValueError("a communication buffer requires an EP process group")

        # Follow the caller's device/default dtype during ordinary construction.
        dtype = dtype or torch.get_default_dtype()
        parameter_device = parameter_device or device or torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            with torch.device(parameter_device):
                self.embed = Embedding(
                    config.vocab_size, config.d_model, std=config.initializer_range
                )
                self.rotary = RotaryEmbedding(
                    config.rotary_width,
                    base=config.rope_base,
                    capacity=config.max_seq_len,
                )
                for index in range(config.n_layers):
                    self.blocks.append(
                        Block(
                            config,
                            index,
                            group=ep_group,
                            buffer=buffer,
                            options=options,
                            device=device,
                            parameter_device=parameter_device,
                        )
                    )
                self.final_norm = RMSNorm(config.d_model, config)
                self.lm_head = Linear(
                    config.d_model, config.vocab_size, std=config.initializer_range
                )
        except BaseException:
            self.close()
            if self._owns_buffer and buffer is not None:
                buffer.destroy()
                self._owns_buffer = False
            raise
        finally:
            torch.set_default_dtype(previous_dtype)

    def expert_parameters(self):
        if self._parallel:
            for block in self.blocks:
                yield from block.moe.experts.expert_parameters()

    def close(self):
        buffer = None
        for block in self.blocks:
            if block.moe.parallel:
                if self._owns_buffer:
                    buffer = block.moe.experts.communication_buffer
                block.moe.experts.close()
        if buffer is not None:
            buffer.destroy()
            self._owns_buffer = False

    def hidden(self, tokens, sequence_lengths=None, *, return_metrics=False):
        c = self.config
        packing = packed_metadata(tokens, sequence_lengths, capacity=c.max_seq_len)
        cumulative, chunks = None, None
        if c.full_attention_interval > 1:
            cumulative, chunks = mlops.prepare_packed_sequence_metadata(
                sequence_lengths if packing.lengths is None else packing.lengths,
                tokens,
            )
        hidden = mlops.embedding(tokens, self.embed.weight)
        counts, probabilities = [], []
        for block in self.blocks:
            hidden, count, probability = block(
                hidden, packing, self.rotary, cumulative, chunks
            )
            counts.append(count)
            probabilities.append(probability)
        counts = torch.stack(counts)
        probabilities = torch.stack(probabilities)
        # Same balancing definition as concatenating all layers' router logits:
        # E * sum_e(mean assignment frequency_e * mean router probability_e).
        total = tokens.numel() * c.n_layers
        # HF sums assignment fractions over the K choices (it does not divide
        # those counts by K); a balanced router therefore has auxiliary ~= K.
        frequency = counts.sum(0).float() / total
        auxiliary = c.n_experts * (frequency * probabilities.sum(0) / total).sum()
        result = self.final_norm(hidden), auxiliary
        if return_metrics:
            return (
                *result,
                {"expert_counts": counts, "probability_sum": probabilities.detach()},
            )
        return result

    def forward(self, tokens, sequence_lengths=None):
        hidden, _ = self.hidden(tokens, sequence_lengths)
        return hidden @ self.lm_head.weight.T

    def loss(
        self,
        tokens,
        targets,
        *,
        seq_lens=None,
        reduction="mean",
        aux_coef=None,
        head_chunk_size=None,
        return_metrics=False,
    ):
        hidden, auxiliary = self.hidden(tokens, seq_lens)
        kwargs = {} if head_chunk_size is None else {"chunk_size": head_chunk_size}
        head = mlops.head_loss(
            hidden, self.lm_head.weight, targets, reduction=reduction, **kwargs
        )
        coefficient = self.config.router_aux_loss_coef if aux_coef is None else aux_coef
        loss = head + coefficient * auxiliary_share(auxiliary, targets, reduction)
        if not return_metrics:
            return loss
        trained = targets.ne(-100).sum()
        return loss, {
            "ce_sum": head.detach() if reduction == "sum" else head.detach() * trained,
            "auxiliary_sum": auxiliary.detach() * trained,
            "trained_tokens": trained,
        }


class Qwen30B(QwenMoE):
    def __init__(self, config=None, **kwargs):
        super().__init__(config or Qwen30BConfig(), **kwargs)


class Qwen35B(QwenMoE):
    def __init__(self, config=None, **kwargs):
        super().__init__(config or Qwen35BConfig(), **kwargs)
