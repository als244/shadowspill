"""Parameter-sharing text decoder with explicit AttnRes and cross-pass message edges."""

import torch
from mlops.attention_residual import attention_residual
from mlops.modules import LanguageModelHead
from torch import nn
from torch.nn import functional as F

from shadowspill_workloads.models.shared.decoder import packed_metadata
from shadowspill_workloads.models.shared.token_head import (
    decoder_target_logprobs,
    decoder_token_features,
)

from ...lora import TARGETS
from .attention import Attention, Norm, ShortConv, rms
from .experts import Experts


def mix(states, partial, query):
    """Attend independently at each token over the normalized residual blocks."""
    values = torch.stack([*(s.float() for s in states), partial.float()], dim=-2)
    return attention_residual(values, query)


class MessageGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.eps = c.norm_epsilon
        self.down_weight = nn.Parameter(torch.empty(c.gate_rank, 2 * c.d_model))
        self.gain_weight = nn.Parameter(torch.empty(c.d_model, c.gate_rank))
        self.forget_weight = nn.Parameter(torch.empty(c.d_model, c.gate_rank))
        for p in self.parameters():
            nn.init.normal_(p, std=c.initializer_range)

    def forward(self, x, msg):
        mixed = torch.cat((rms(x, self.eps), rms(msg, self.eps)), dim=-1)
        low = F.linear(mixed.to(self.down_weight.dtype), self.down_weight).float()
        forget = F.linear(low, self.forget_weight.float()).tanh()
        gain = F.linear(low, self.gain_weight.float()).tanh()
        return (1 - forget) * x.float() + gain * msg


class Layer(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.local = i in c.local_layer_ids
        self.boundary = i % c.block_size == 0
        query_shape = (c.d_model,) if self.local else (3, c.d_model)
        self.attn_query = (
            nn.Parameter(torch.zeros(query_shape, dtype=torch.float32)) if i else None
        )
        self.mlp_query = nn.Parameter(torch.zeros(c.d_model, dtype=torch.float32))
        self.attn = Attention(c, self.local)
        self.mlp = Experts(c, i)
        self.mlp_norm = Norm(c)
        if self.local:
            self.attn_norm = Norm(c)
        else:
            self.q_input_norm, self.k_input_norm, self.v_input_norm = (
                Norm(c),
                Norm(c),
                Norm(c),
            )
        if c.core_start <= i < c.core_end:
            self.controllers = nn.ModuleDict(
                {name: MessageGate(c) for name in ("attn", "k", "v", "mlp")}
            )
        self.attn_sconv = ShortConv(c.d_model, c.conv_kernel)
        self.mlp_sconv = ShortConv(c.d_model, c.conv_kernel)

    def forward(self, x, blocks, positions, distance, previous=None, queries=None):
        q = k = v = x
        if self.attn_query is not None:
            values = mix(blocks, x, self.attn_query)
            if self.local:
                q = k = v = values.squeeze(-2)
            else:
                q, k, v = values.unbind(-2)
        if previous is not None:
            messages = mix(*previous, queries).unbind(-2)
            q = self.controllers["attn"](q, messages[0])
            k = self.controllers["k"](k, messages[1])
            v = self.controllers["v"](v, messages[2])
        if self.local:
            q = self.attn_norm(q, x.dtype)
            if previous is None:
                k = v = q
            else:
                k, v = self.attn_norm(k, x.dtype), self.attn_norm(v, x.dtype)
        else:
            q, k, v = (
                self.q_input_norm(q, x.dtype),
                self.k_input_norm(k, x.dtype),
                self.v_input_norm(v, x.dtype),
            )
        attn = self.attn_sconv(
            self.attn(q, k, v, positions, distance), positions
        ).float()
        if self.boundary:
            blocks = (*blocks, x)
            partial = attn
        else:
            partial = x.float() + attn
        value = mix(blocks, partial, self.mlp_query).squeeze(-2)
        if previous is not None:
            value = self.controllers["mlp"](value, messages[3])
        value = self.mlp(self.mlp_norm(value, x.dtype))
        return (partial + self.mlp_sconv(value, positions).float()).to(x.dtype), blocks


class Embedding(nn.Embedding):
    def __init__(self, c):
        super().__init__(c.embedding_vocab_size, c.d_model)
        self.embed_norm = Norm(c)

    def forward(self, tokens):
        return self.embed_norm(super().forward(tokens))


class Head(LanguageModelHead):
    def __init__(self, c):
        self.initializer_range = c.initializer_range
        super().__init__(
            c.d_model,
            c.vocab_size,
            fp32_projection=c.head_fp32_projection,
            logit_softcap=c.logit_cap,
        )

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.initializer_range)


class Model(nn.Module):
    LORA_TARGETS = TARGETS
    token_features = decoder_token_features
    target_logprobs = decoder_target_logprobs

    def __init__(self, config):
        super().__init__()
        c = self.config = config
        self.embed = Embedding(c)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.n_layers))
        rows = 2 * (c.core_end - c.core_start)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(rows, c.d_model, dtype=torch.float32)
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(rows, c.d_model, dtype=torch.float32)
        )
        self.output_query = nn.Parameter(torch.zeros(c.d_model, dtype=torch.float32))
        self.norm = Norm(c)
        self.lm_head = Head(c)

    def hidden(self, tokens, sequence_lengths=None):
        c = self.config
        positions = packed_metadata(
            tokens, sequence_lengths, capacity=c.max_seq_len
        ).positions.reshape_as(tokens)
        distance = None
        if not c.attention_query_chunk_size and not c.use_fused_attention(
            tokens.shape[1]
        ):
            index = torch.arange(tokens.shape[1], device=tokens.device)
            distance = index[:, None] - index[None, :]
        x, blocks = self.embed(tokens), ()
        for i in range(c.core_start):
            x, blocks = self.layers[i](x, blocks, positions, distance)
        prefix, prefix_blocks = x, blocks
        previous = None
        for p in range(c.passes):
            if p:
                previous = (blocks[len(prefix_blocks) + 1 :], x)
                x, blocks = prefix, prefix_blocks
            for i in range(c.core_start, c.core_end):
                queries = None
                if previous is not None:
                    j = 2 * (i - c.core_start)
                    qm = self.middle_cycle_cross_pass_queries
                    kv = self.middle_cycle_cross_pass_kv_queries
                    queries = torch.stack((qm[j], kv[j], kv[j + 1], qm[j + 1]))
                x, blocks = self.layers[i](
                    x, blocks, positions, distance, previous, queries
                )
        for i in range(c.core_end, c.n_layers):
            x, blocks = self.layers[i](x, blocks, positions, distance)
        return self.norm(mix(blocks, x, self.output_query).squeeze(-2), x.dtype)

    def forward(self, tokens, sequence_lengths=None):
        return self.lm_head(self.hidden(tokens, sequence_lengths))

    def loss(
        self, tokens, targets, seq_lens=None, *, reduction="mean", head_chunk_size=None
    ):
        if reduction == "none":
            return -self.target_logprobs(
                tokens, targets, seq_lens, head_chunk_size=head_chunk_size
            ).reshape(-1)
        if reduction not in ("mean", "sum"):
            raise ValueError(f"unsupported cross-entropy reduction {reduction!r}")
        valid = targets != -100
        weights = torch.ones_like(targets, dtype=torch.float32)
        if reduction == "mean":
            count = valid.sum()
            weights = weights / count.clamp_min(1)
        loss, _ = self.lm_head.objective(
            self.hidden(tokens, seq_lens),
            targets,
            objective="cross_entropy",
            inputs=(valid, weights),
            chunk_size=head_chunk_size,
        )
        if reduction == "sum":
            return loss
        return torch.where(count > 0, loss, float("nan"))
