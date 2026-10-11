"""Packed KDA and NoPE sparse MLA; all trainable projections stay in the model."""

import mlops
import torch
from mlops import sparse_latent_attention
from mlops import gated_rms_norm, kda_decay
from mlops import pooled_topk
from mlops import kimi_delta_attention
from mlops.lora import LoRALinear
from torch import nn
from torch.nn import functional as F

from .common import ForgetGate, RMSNorm, linear


class GatedNorm(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(config.linear_head_dim, device=device, dtype=config.dtype)
        )
        self.eps = config.rms_norm_eps

    def forward(self, x, gate):
        return gated_rms_norm(x, gate, self.weight, eps=self.eps, activation="sigmoid")


class KDA(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        c = self.config = config
        width = c.linear_num_heads * c.linear_head_dim
        self.q_proj = linear(c.hidden_size, width, c, device)
        self.k_proj = linear(c.hidden_size, width, c, device)
        self.v_proj = linear(c.hidden_size, width, c, device)
        # HF's keep-in-FP32 storage rule applies to the convolution.
        self.conv1d = nn.Conv1d(
            3 * width,
            3 * width,
            c.linear_conv_kernel_dim,
            groups=3 * width,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        nn.init.normal_(self.conv1d.weight, std=c.initializer_range)
        self.forget_gate = ForgetGate(c, device)
        self.b_proj = linear(c.hidden_size, c.linear_num_heads, c, device)
        self.g_a_proj = linear(c.hidden_size, c.linear_head_dim, c, device)
        self.g_b_proj = linear(c.linear_head_dim, width, c, device)
        self.o_norm = GatedNorm(c, device)
        self.o_proj = linear(width, c.hidden_size, c, device)

    def forward(self, x, boundaries, cumulative, chunks):
        c = self.config
        packed = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), -1)
        packed = mlops.causal_conv_silu(packed, self.conv1d.weight, cumulative, chunks)
        q, k, v = (
            t.reshape(-1, c.linear_num_heads, c.linear_head_dim)
            for t in packed.chunk(3, -1)
        )
        q, k = mlops.l2_norm(q), mlops.l2_norm(k)
        raw = self.forget_gate.f_b_proj(self.forget_gate.f_a_proj(x)).reshape_as(q)
        g = kda_decay(
            raw,
            self.forget_gate.dt_bias,
            self.forget_gate.A_log,
            lower_bound=c.linear_lower_bound,
        )
        attended = kimi_delta_attention(
            q, k, v, g, self.b_proj(x).sigmoid(), cumulative, chunks
        )
        gate = self.g_b_proj(self.g_a_proj(x)).reshape_as(attended)
        return self.o_proj(self.o_norm(attended, gate).flatten(-2))


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
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(c.index_kpool, c.index_head_dim, device=device, dtype=c.dtype)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.ones(c.index_head_dim, c.hidden_size, device=device, dtype=c.dtype)
        )
        self.requires_grad_(False)  # HF's selection stage is explicitly no_grad.

    @torch.no_grad()
    def forward(self, x, q_residual, boundaries):
        c = self.config
        q = self.wq_b(q_residual).reshape(-1, c.index_n_heads, c.index_head_dim)
        key = self.k_norm(self.wk(x))
        gates = F.linear(x, self.index_kpool_compress_gate)
        weights = self.weights_proj(x).float()
        return pooled_topk(
            q,
            key,
            gates,
            weights,
            self.index_kpool_compress_ape,
            boundaries,
            top_k=c.index_topk,
        )


class SparseMLA(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        c = self.config = config
        self.q_a_proj = linear(c.hidden_size, c.q_lora_rank, c, device)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, c, device)
        self.q_b_proj = linear(
            c.q_lora_rank, c.num_attention_heads * c.qk_nope_head_dim, c, device
        )
        self.kv_a_proj_with_mqa = linear(c.hidden_size, c.kv_lora_rank, c, device)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, c, device)
        self.kv_b_proj = linear(
            c.kv_lora_rank,
            c.num_attention_heads * (c.qk_nope_head_dim + c.v_head_dim),
            c,
            device,
        )
        self.o_proj = linear(
            c.num_attention_heads * c.v_head_dim, c.hidden_size, c, device
        )
        self.indexer = Indexer(c, device)

    def forward(self, x, boundaries, cumulative, chunks):
        c = self.config
        q_residual = self.q_a_layernorm(self.q_a_proj(x))
        query = self.q_b_proj(q_residual).reshape(
            -1, c.num_attention_heads, c.qk_nope_head_dim
        )
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        indices = self.indexer(x, q_residual, boundaries)
        wk, wv = self.kv_b_proj.weight.reshape(
            c.num_attention_heads, -1, c.kv_lora_rank
        ).split((c.qk_nope_head_dim, c.v_head_dim), dim=1)
        q_latent = torch.einsum("thd,hdl->thl", query, wk)
        if isinstance(self.kv_b_proj, LoRALinear):
            a = self.kv_b_proj.lora_a.to(x.dtype)
            bk, bv = (
                self.kv_b_proj.lora_b.to(x.dtype)
                .reshape(c.num_attention_heads, -1, a.shape[0])
                .split((c.qk_nope_head_dim, c.v_head_dim), dim=1)
            )
            scale = self.kv_b_proj.lora_config.scale
            q_latent = q_latent + scale * F.linear(
                torch.einsum("thd,hdr->thr", query, bk), a.T
            )
        attended = sparse_latent_attention(
            q_latent, latent, indices, scale=c.qk_nope_head_dim**-0.5
        )
        value = torch.einsum("thl,hdl->thd", attended, wv)
        if isinstance(self.kv_b_proj, LoRALinear):
            value = value + scale * torch.einsum(
                "thr,hdr->thd", F.linear(attended, a), bv
            )
        return self.o_proj(value.flatten(-2))
