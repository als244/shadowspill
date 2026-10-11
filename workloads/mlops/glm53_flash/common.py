"""Ordinary model parameters and projection composition around stateless ops."""

import math
from contextlib import nullcontext

import torch
from mlops import hyper_connection as hc
from torch import nn
from torch.nn import functional as F


def without_autocast(value):
    """Suppress implicit casts; avoid redundant regions during whole-model export.

    The caller still chooses tensor dtypes explicitly.
    """
    device_type = value.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def linear(inputs, outputs, config, device):
    module = nn.Linear(inputs, outputs, bias=False, device=device, dtype=config.dtype)
    nn.init.normal_(module.weight, std=config.initializer_range)
    return module


class RMSNorm(nn.Module):
    def __init__(self, width, config, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=config.dtype))
        self.eps = config.rms_norm_eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(x.dtype)


class HyperConnection(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        n, d = config.hc_mult, config.hidden_size
        self.fn = nn.Parameter(
            torch.empty(n * (n + 2), n * d, device=device, dtype=config.dtype)
        )
        self.base = nn.Parameter(
            torch.zeros(n * (n + 2), device=device, dtype=torch.float32)
        )
        self.scale = nn.Parameter(torch.ones(3, device=device, dtype=torch.float32))
        nn.init.normal_(self.fn, std=config.initializer_range)

    def forward(self, streams):
        c = self.config
        # FP32 projection must not be downcast by an enclosing autocast context.
        with without_autocast(streams):
            x = hc.mhc_normalize_streams(streams, eps=c.rms_norm_eps)
            projected = F.linear(x, self.fn.float())
            return hc.mhc_coefficients(
                streams,
                projected,
                self.base,
                self.scale,
                sinkhorn_eps=c.hc_eps,
                iterations=c.hc_sinkhorn_iters,
            )


class ForgetGate(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        c = config
        self.f_a_proj = linear(c.hidden_size, c.linear_head_dim, c, device)
        self.f_b_proj = linear(
            c.linear_head_dim, c.linear_num_heads * c.linear_head_dim, c, device
        )
        self.dt_bias = nn.Parameter(
            torch.empty(
                c.linear_num_heads * c.linear_head_dim,
                device=device,
                dtype=torch.float32,
            )
        )
        self.A_log = nn.Parameter(
            torch.zeros(c.linear_num_heads, device=device, dtype=torch.float32)
        )
        with torch.no_grad():
            dt = (
                self.dt_bias.uniform_(math.log(1e-3), math.log(1e-1))
                .exp()
                .clamp_min(1e-4)
            )
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
