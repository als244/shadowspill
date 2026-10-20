"""GLM-5.3 language backbone: ordinary residuals and cross-layer DSA indices."""

import torch
from mlops.modules import LanguageModelHead
from torch import nn

from workloads.models.shared.glm.common import RMSNorm
from workloads.models.shared.glm.lora import enable_lora
from workloads.models.shared.glm.moe import DenseMLP, MoE

from .attention import SparseMLA, positions_for


class DecoderLayer(nn.Module):
    def __init__(self, config, index, device):
        super().__init__()
        self.self_attn = SparseMLA(config, index, device)
        self.mlp = (DenseMLP if index < config.first_k_dense_replace else MoE)(
            config, device
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config, device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config, device)

    def forward(self, x, boundaries, positions, indices):
        branch, indices = self.self_attn(
            self.input_layernorm(x), boundaries, positions, indices
        )
        x = x + branch
        return x + self.mlp(self.post_attention_layernorm(x)), indices


class Model(nn.Module):
    def __init__(self, config, *, device="cuda", vision_config=None):
        super().__init__()
        if vision_config is not None:
            raise ValueError("GLM-5.3 is a text-only workload")
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=config.dtype
        )
        nn.init.normal_(self.embed_tokens.weight, std=config.initializer_range)
        self.layers = nn.ModuleList(
            DecoderLayer(config, i, device) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config, device)
        self.lm_head = LanguageModelHead(
            config.hidden_size, config.vocab_size, device=device, dtype=config.dtype
        )
        nn.init.normal_(self.lm_head.weight, std=config.initializer_range)
        if config.lora_rank:
            enable_lora(self, config)

    def hidden(self, tokens, boundaries, cumulative=None, chunks=None):
        if isinstance(boundaries, torch.Tensor):
            raise ValueError(
                "GLM-5.3 expects static packed sequence boundaries as a tuple"
            )
        if tokens.ndim != 1 or boundaries[0] != 0 or boundaries[-1] != tokens.numel():
            raise ValueError(
                "Packed sequence boundaries must cover the flat token input"
            )
        x = self.embed_tokens(tokens)
        positions = positions_for(boundaries, x)
        indices = None
        for layer in self.layers:
            x, indices = layer(x, boundaries, positions, indices)
        return self.norm(x)

    def forward(self, tokens, boundaries, cumulative=None, chunks=None):
        return self.lm_head(self.hidden(tokens, boundaries, cumulative, chunks))

    def loss(
        self,
        tokens,
        targets,
        boundaries,
        cumulative=None,
        chunks=None,
        *,
        chunk_size=None,
        valid_rows=None,
        reduction="mean",
    ):
        return self.lm_head.loss(
            self.hidden(tokens, boundaries, cumulative, chunks),
            targets,
            chunk_size=chunk_size,
            valid_rows=valid_rows,
            reduction=reduction,
        )
