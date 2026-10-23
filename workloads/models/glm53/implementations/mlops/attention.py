"""Packed positional sparse MLA with the checkpoint's cross-layer top-k reuse."""

from itertools import pairwise

import torch
from mlops import pooled_topk, sparse_latent_attention
from mlops.lora import LoRALinear
from mlops.sequential_moe.precision.linear import FrozenLinear, FrozenLoRALinear
from torch import nn
from torch.nn import functional as F

from workloads.models.shared.glm.common import RMSNorm, linear


def positions_for(boundaries, x):
    return torch.cat(
        [
            torch.arange(end - start, device=x.device)
            for start, end in pairwise(boundaries)
        ]
    )


def rotate(x, positions, theta):
    """Interleaved input pairs, emitted as real/imaginary halves like HF GLM."""
    width = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angles = positions.float()[:, None] * frequencies
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    if x.ndim == 3:
        cosine, sine = cosine[:, None], sine[:, None]
    real, imaginary = x[..., 0::2], x[..., 1::2]
    return torch.cat(
        (real * cosine - imaginary * sine, imaginary * cosine + real * sine), -1
    )


class Indexer(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        c = self.config = config
        self.wq_b = linear(c.q_lora_rank, c.index_n_heads * c.index_head_dim, c, device)
        self.wk = linear(c.hidden_size, c.index_head_dim, c, device)
        self.k_norm = nn.LayerNorm(
            c.index_head_dim, eps=1e-6, device=device, dtype=c.dtype
        )
        self.weights_proj = linear(c.hidden_size, c.index_n_heads, c, device)
        self.requires_grad_(False)

    def forward(self, x, q_residual, boundaries, positions):
        c = self.config
        q = self.wq_b(q_residual).reshape(-1, c.index_n_heads, c.index_head_dim)
        key = self.k_norm(self.wk(x))
        r = c.qk_rope_head_dim
        q = torch.cat((rotate(q[..., :r], positions, c.rope_theta), q[..., r:]), -1)
        key = torch.cat(
            (rotate(key[..., :r], positions, c.rope_theta), key[..., r:]), -1
        )
        if not isinstance(boundaries, torch.Tensor):
            boundaries = torch.tensor(boundaries, dtype=torch.int64, device="cpu")
        # Pool size one is ordinary causal token selection: no key compression
        # and no forced tail. Reuse the bounded Triton scorer/selector, with
        # BF16 dot-product operands and FP32 score accumulation and head weights.
        return pooled_topk(
            q,
            key,
            torch.zeros_like(key),
            self.weights_proj(x).float(),
            key.new_zeros((1, c.index_head_dim)),
            boundaries,
            top_k=c.index_topk,
        )


class SparseMLA(nn.Module):
    def __init__(self, config, index, device):
        super().__init__()
        c = self.config = config
        self.q_a_proj = linear(c.hidden_size, c.q_lora_rank, c, device)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, c, device)
        self.q_a_layernorm.eps = 1e-6
        self.q_b_proj = linear(
            c.q_lora_rank,
            c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim),
            c,
            device,
        )
        self.kv_a_proj_with_mqa = linear(
            c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim, c, device
        )
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, c, device)
        self.kv_a_layernorm.eps = 1e-6
        self.kv_b_proj = linear(
            c.kv_lora_rank,
            c.num_attention_heads * (c.qk_nope_head_dim + c.v_head_dim),
            c,
            device,
        )
        self.o_proj = linear(
            c.num_attention_heads * c.v_head_dim, c.hidden_size, c, device
        )
        self.indexer = Indexer(c, device) if c.indexer_types[index] == "full" else None

    def forward(self, x, boundaries, positions, previous_indices=None):
        c = self.config
        q_residual = self.q_a_layernorm(self.q_a_proj(x))
        query, q_rope = (
            self.q_b_proj(q_residual)
            .reshape(-1, c.num_attention_heads, c.qk_nope_head_dim + c.qk_rope_head_dim)
            .split((c.qk_nope_head_dim, c.qk_rope_head_dim), -1)
        )
        latent, k_rope = self.kv_a_proj_with_mqa(x).split(
            (c.kv_lora_rank, c.qk_rope_head_dim), -1
        )
        latent = self.kv_a_layernorm(latent)
        q_rope = rotate(q_rope, positions, c.rope_theta)
        k_rope = rotate(k_rope, positions, c.rope_theta)
        indices = (
            previous_indices
            if self.indexer is None
            else self.indexer(x, q_residual, boundaries, positions)
        )
        if indices is None:
            raise ValueError(
                "Shared sparse attention requires a preceding full indexer"
            )
        weight = (
            self.kv_b_proj.materialize_weight()
            if isinstance(self.kv_b_proj, FrozenLinear)
            else self.kv_b_proj.weight
        )
        wk, wv = weight.reshape(c.num_attention_heads, -1, c.kv_lora_rank).split(
            (c.qk_nope_head_dim, c.v_head_dim), 1
        )
        q_latent = torch.einsum("thd,hdl->thl", query, wk)
        lora = isinstance(self.kv_b_proj, LoRALinear | FrozenLoRALinear)
        if lora:
            a = self.kv_b_proj.lora_a.to(x.dtype)
            bk, bv = (
                self.kv_b_proj.lora_b.to(x.dtype)
                .reshape(c.num_attention_heads, -1, a.shape[0])
                .split((c.qk_nope_head_dim, c.v_head_dim), 1)
            )
            scale = self.kv_b_proj.lora_config.scale
            update = F.linear(torch.einsum("thd,hdr->thr", query, bk), a.T)
            if scale != 1.0:
                update = scale * update
            q_latent = q_latent + update
        attended = sparse_latent_attention(
            q_latent,
            latent,
            indices,
            scale=(c.qk_nope_head_dim + c.qk_rope_head_dim) ** -0.5,
            q_rope=q_rope,
            k_rope=k_rope,
        )
        value = torch.einsum("thl,hdl->thd", attended, wv)
        if lora:
            update = torch.einsum("thr,hdr->thd", F.linear(attended, a), bv)
            if scale != 1.0:
                update = scale * update
            value = value + update
        return self.o_proj(value.flatten(-2)), indices
