"""GLM's MoE on ``mlops.moe``, or ``expert_lora`` when the experts carry LoRA.

GLM routes by sigmoid scores: the correction bias (a buffer, never trained by
gradient) only selects, the selected scores are normalized and scaled by
``routed_scaling_factor``. Every expert's SwiGLU, the shared expert's too, is
clamped at ``swiglu_limit``. The routed experts are one stacked bank in
``nn.Linear`` orientation, ``w13 [E, 2I, H]`` (each expert's gate rows, then
its up rows) and ``w2 [E, H, I]``, so the façade's selection decides the
kernels: SonicMoE on Hopper and Blackwell, the builtin grouped GEMMs
elsewhere.
"""

from dataclasses import replace

import mlops
import torch
from mlops import packed_swiglu
from mlops.lora import LoRAConfig
from mlops.lora.experts import expert_lora
from torch import nn

from ..routed_lora import add_factors, factors
from .common import linear, without_autocast


class DenseMLP(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        c = self.config = config
        self.gate_proj = linear(c.hidden_size, c.intermediate_size, c, device)
        self.up_proj = linear(c.hidden_size, c.intermediate_size, c, device)
        self.down_proj = linear(c.intermediate_size, c.hidden_size, c, device)

    def forward(self, x):
        packed = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(packed_swiglu(packed, limit=self.config.swiglu_limit))


def expert_weight(shape, config, device, *, gate_up):
    """One stacked expert weight ``[E, out, in]``.

    ``expert_storage="bf16"`` gives a trainable bf16 parameter. ``"fp8"`` gives
    a frozen `QuackFP8Weight`: both E4M3 orientations with a descale per row,
    the gate and up rows of ``w13`` interleaved as Quack's SwiGLU epilogues
    read them. ``"fp8_block"`` gives a frozen `BlockFP8Weight`: E4M3 once with
    a scale per 128x128 block, rows in ``nn.Linear`` order. Either way
    ``mlops.moe`` computes the experts' GEMMs in FP8 (SM90). On the meta
    device only the components are declared, for a checkpoint to fill.
    """
    if config.expert_storage == "bf16":
        value = torch.empty(shape, device=device, dtype=config.dtype)
        nn.init.normal_(value, std=config.initializer_range)
        return nn.Parameter(value)
    experts, out, inner = shape
    meta = torch.device(device).type == "meta"
    if config.expert_storage == "fp8_block":
        from mlops.block_fp8 import BlockFP8Weight

        if meta:
            blocks = (experts, -(-out // 128), -(-inner // 128))
            container = BlockFP8Weight(
                torch.empty(shape, device=device, dtype=torch.float8_e4m3fn),
                torch.empty(blocks, device=device, dtype=torch.float32),
                dtype=config.dtype,
            )
        else:
            value = torch.empty(shape, device=device, dtype=config.dtype)
            nn.init.normal_(value, std=config.initializer_range)
            container = BlockFP8Weight.quantize(value)
        return nn.Parameter(container, requires_grad=False)
    from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight

    if meta:
        fp8 = {"device": device, "dtype": torch.float8_e4m3fn}
        scales = {"device": device, "dtype": torch.float32}
        container = QuackFP8Weight(
            torch.empty(shape, **fp8),
            torch.empty((experts, inner, out), **fp8),
            torch.empty((experts, out), **scales),
            torch.empty((experts, inner), **scales),
            dtype=config.dtype,
            interleaved=gate_up,
        )
    else:
        value = torch.empty(shape, device=device, dtype=config.dtype)
        nn.init.normal_(value, std=config.initializer_range)
        container = QuackFP8Weight.quantize(value, interleaved=gate_up)
    return nn.Parameter(container, requires_grad=False)


class MoE(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        c = self.config = config
        experts, hidden, width = (
            c.num_local_experts,
            c.hidden_size,
            c.moe_intermediate_size,
        )
        router = torch.empty(
            experts, hidden, device=device, dtype=c.router_weight_dtype
        )
        nn.init.normal_(router, std=c.initializer_range)
        self.router = nn.Parameter(router)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(experts, device=device, dtype=torch.float32),
        )
        gate_up, down = self.expert_shapes
        self.w13 = expert_weight(gate_up, c, device, gate_up=True)
        self.w2 = expert_weight(down, c, device, gate_up=False)
        self.shared_experts = (
            DenseMLP(replace(c, intermediate_size=width * c.n_shared_experts), device)
            if c.n_shared_experts
            else None
        )
        if c.lora_rank:
            add_factors(
                self,
                LoRAConfig(
                    rank=c.lora_rank,
                    alpha=c.lora_alpha,
                    factor_dtype=c.lora_factor_dtype,
                ),
            )

    @property
    def expert_shapes(self):
        """The routed experts' gate/up and down banks, ``[E, out, in]``."""
        c = self.config
        experts, hidden, width = (
            c.num_local_experts,
            c.hidden_size,
            c.moe_intermediate_size,
        )
        return (experts, 2 * width, hidden), (experts, hidden, width)

    def forward(self, x):
        c = self.config
        # The shared expert's output is the base the routed experts add to.
        base = (
            self.shared_experts(x)
            if self.shared_experts is not None
            else torch.zeros_like(x)
        )
        options = {
            "top_k": c.num_experts_per_tok,
            "routing_mode": "sigmoid_noaux_tc",
            "router_bias": self.e_score_correction_bias,
            "routed_scaling": c.routed_scaling_factor,
            "swiglu_limit": c.swiglu_limit,
        }
        with without_autocast(x):
            lora = getattr(self, "lora_config", None)
            if lora is None:
                output, *_ = mlops.moe(
                    x, base, self.router, self.w13, self.w2, **options
                )
            else:
                output, *_ = expert_lora(
                    x,
                    base,
                    self.router,
                    self.w13,
                    self.w2,
                    *factors(self),
                    scale=lora.scale,
                    **options,
                )
        return output
