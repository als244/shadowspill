"""GLM-5.3-Flash language model with optional vision, composed from MLOps ops."""

import torch
from mlops import mhc_combine
from mlops.modules import LanguageModelHead
from torch import nn

from workloads.models.glm53_flash.lora import SHARED_TARGETS as LORA_SHARED_TARGETS
from workloads.models.glm53_flash.lora import TARGETS as LORA_TARGETS
from workloads.models.shared.glm.common import HyperConnection, RMSNorm
from workloads.models.shared.glm.lora import enable_lora
from workloads.models.shared.glm.moe import DenseMLP, MoE

from .attention import KDA, SparseMLA
from .vision import VisionModel


class DecoderLayer(nn.Module):
    def __init__(self, config, index, device):
        super().__init__()
        self.self_attn = (
            KDA if config.layer_types[index] == "linear_attention" else SparseMLA
        )(config, device)
        self.mlp = (DenseMLP if index < config.first_k_dense_replace else MoE)(
            config, device
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config, device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config, device)
        self.hc_coefficient_precision = config.hc_coefficient_precision
        self.attn_hc = HyperConnection(config, device)
        self.ffn_hc = HyperConnection(config, device)

    def forward(self, streams, boundaries, cumulative, chunks):
        post, mixing, x = self.attn_hc(streams)
        branch = self.self_attn(self.input_layernorm(x), boundaries, cumulative, chunks)
        streams = mhc_combine(
            branch,
            streams,
            post,
            mixing,
            coefficient_precision=self.hc_coefficient_precision,
        )
        post, mixing, x = self.ffn_hc(streams)
        branch = self.mlp(self.post_attention_layernorm(x))
        return mhc_combine(
            branch,
            streams,
            post,
            mixing,
            coefficient_precision=self.hc_coefficient_precision,
        )


class Model(nn.Module):
    LORA_TARGETS = LORA_TARGETS
    LORA_SHARED_TARGETS = LORA_SHARED_TARGETS

    def __init__(self, config, *, device="cuda", vision_config=None):
        super().__init__()
        self.config = config
        if (
            vision_config is not None
            and vision_config.out_hidden_size != config.hidden_size
        ):
            raise ValueError("Vision output width must match the language-model width")
        self.visual = (
            None if vision_config is None else VisionModel(vision_config, device=device)
        )
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

    def hidden(self, tokens, boundaries, cumulative, chunks, images=None):
        if not isinstance(boundaries, torch.Tensor):
            boundaries = torch.tensor(boundaries, dtype=torch.int64, device="cpu")
        if tokens.ndim != 1:
            raise ValueError(
                "GLM expects flattened token IDs and explicit sequence boundaries"
            )
        image_features = None
        if images is not None:
            if self.visual is None:
                raise ValueError(
                    "Build with a vision_config before passing image inputs"
                )
            image_features = self.visual(
                images.pixels, images.positions, images.boundaries
            )
        hidden = self.embed_tokens(tokens)
        if image_features is not None:
            if image_features.shape[0] != images.token_indices.numel():
                raise ValueError(
                    "Image features and token positions have different counts"
                )
            hidden = hidden.index_copy(
                0, images.token_indices, image_features.to(hidden.dtype)
            )
        streams = hidden.unsqueeze(-2).expand(-1, self.config.hc_mult, -1).contiguous()
        for layer in self.layers:
            streams = layer(streams, boundaries, cumulative, chunks)
        # Published hyper-head: unweighted stream mean, then final RMS normalization.
        return self.norm(streams.mean(-2))

    def forward(self, tokens, boundaries, cumulative, chunks, images=None):
        return self.lm_head(self.hidden(tokens, boundaries, cumulative, chunks, images))

    def loss(
        self,
        tokens,
        targets,
        boundaries,
        cumulative,
        chunks,
        *,
        images=None,
        chunk_size=None,
        valid_rows=None,
        reduction="mean",
    ):
        """Cross entropy with bounded logits; use sum for microbatch accumulation."""
        return self.lm_head.loss(
            self.hidden(tokens, boundaries, cumulative, chunks, images),
            targets,
            chunk_size=chunk_size,
            valid_rows=valid_rows,
            reduction=reduction,
        )
