"""Packed causal convolution, gates and the reference DeltaNet recurrence."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from workloads.models.shared.decoder import GatedRMSNorm, l2_normalize

from ...config import Config
from .delta_rule.operation import _delta_rule


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.w_qkvz = nn.Linear(config.d_model, config.qkvz_width, bias=False)
        self.w_ba = nn.Linear(config.d_model, 2 * config.lin_v_heads, bias=False)
        self.conv = nn.Conv1d(
            config.convolution_width,
            config.convolution_width,
            config.lin_conv_kernel,
            groups=config.convolution_width,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.empty(config.lin_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(config.lin_v_heads))
        self.lin_norm = GatedRMSNorm(config.lin_v_head_dim, epsilon=config.norm_epsilon)
        self.w_out = nn.Linear(config.linear_value_width, config.d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the parameters this module owns, in place."""

        with torch.no_grad():
            self.A_log.zero_()
            self.dt_bias.zero_()

    def _segment(self, hidden: torch.Tensor) -> torch.Tensor:
        config = self.config
        batch, sequence, _width = hidden.shape
        qkvz = self.w_qkvz(hidden)
        controls = self.w_ba(hidden)
        convolution_input = qkvz[..., : config.convolution_width]
        padded = F.pad(
            convolution_input.float().transpose(1, 2),
            (config.lin_conv_kernel - 1, 0),
        )
        convolved = F.conv1d(
            padded, self.conv.weight.float(), groups=config.convolution_width
        )
        convolved = F.silu(convolved.transpose(1, 2)).to(hidden.dtype)
        query = l2_normalize(
            convolved[..., : config.linear_key_width].view(
                batch, sequence, config.lin_k_heads, config.lin_k_head_dim
            )
        )
        key = l2_normalize(
            convolved[..., config.linear_key_width : 2 * config.linear_key_width].view(
                batch, sequence, config.lin_k_heads, config.lin_k_head_dim
            )
        )
        value = convolved[..., 2 * config.linear_key_width :].view(
            batch, sequence, config.lin_v_heads, config.lin_v_head_dim
        )
        beta = torch.sigmoid(controls[..., : config.lin_v_heads].float()).to(
            hidden.dtype
        )
        raw_decay = controls[..., config.lin_v_heads :].float()
        decay = -self.A_log.float().exp() * F.softplus(raw_decay + self.dt_bias)
        mixed = _delta_rule(query, key, value, beta, decay)
        gate = qkvz[..., config.convolution_width :].view_as(mixed)
        normalized = self.lin_norm(mixed, gate)
        return self.w_out(normalized.reshape(batch, sequence, -1))

    def forward(self, hidden: torch.Tensor, lengths: tuple[int, ...]) -> torch.Tensor:
        batch, sequence, _width = hidden.shape
        if len(lengths) == batch and len(set(lengths)) == 1:
            return self._segment(hidden)
        flat = hidden.reshape(1, batch * sequence, -1)
        outputs: list[torch.Tensor] = []
        offset = 0
        for length in lengths:
            outputs.append(self._segment(flat[:, offset : offset + length]))
            offset += length
        return torch.cat(outputs, dim=1).view_as(hidden)
