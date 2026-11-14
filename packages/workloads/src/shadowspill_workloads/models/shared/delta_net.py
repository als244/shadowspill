"""The Gated DeltaNet sequence mixer on mlops operations, shared by the dense
Qwen 3.5 model and the Qwen mixture families."""

from __future__ import annotations

import mlops
import torch
import torch.nn as nn

from .norms import GatedRMSNorm


class GatedDeltaNet(nn.Module):
    def __init__(self, config) -> None:
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
        self.lin_norm = GatedRMSNorm(
            config.lin_v_head_dim, epsilon=getattr(config, "norm_epsilon", 1e-5)
        )
        self.w_out = nn.Linear(config.linear_value_width, config.d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the parameters this module owns, in place."""

        with torch.no_grad():
            self.A_log.zero_()
            self.dt_bias.zero_()

    def forward(
        self,
        hidden: torch.Tensor,
        cumulative: torch.Tensor,
        chunk_indices: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        batch, sequence, _width = hidden.shape
        qkvz = self.w_qkvz(hidden)
        controls = self.w_ba(hidden)
        convolved = mlops.causal_conv_silu(
            qkvz[..., : config.convolution_width],
            self.conv.weight,
            cumulative,
            chunk_indices,
        )
        query = mlops.l2_norm(
            convolved[..., : config.linear_key_width].reshape(
                batch * sequence, config.lin_k_heads, config.lin_k_head_dim
            )
        )
        key = mlops.l2_norm(
            convolved[
                ..., config.linear_key_width : 2 * config.linear_key_width
            ].reshape(batch * sequence, config.lin_k_heads, config.lin_k_head_dim)
        )
        value = convolved[..., 2 * config.linear_key_width :].reshape(
            batch * sequence, config.lin_v_heads, config.lin_v_head_dim
        )
        beta = torch.sigmoid(controls[..., : config.lin_v_heads].float()).to(
            hidden.dtype
        )
        decay = controls[..., config.lin_v_heads :]
        mixed = mlops.linear_attention(
            query,
            key,
            value,
            beta.reshape(batch * sequence, config.lin_v_heads),
            decay.reshape(batch * sequence, config.lin_v_heads),
            self.A_log,
            self.dt_bias,
            cumulative,
            chunk_indices,
        ).reshape(batch, sequence, config.lin_v_heads, config.lin_v_head_dim)
        gate = qkvz[..., config.convolution_width :].view_as(mixed)
        normalized = self.lin_norm(mixed, gate)
        return self.w_out(normalized.reshape(batch, sequence, -1))
