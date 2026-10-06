"""OLMoE decoder with MLOps attention and QuackMoE expert parallelism.

The caller supplies the process group and token capacity or existing buffers.
This module knows nothing about planning or training backends; it is a PyTorch model
with fixed-device communication resources, like its QuackMoE submodules.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from mlops.expert_parallel import QuackMoE
from mlops.expert_parallel import QuackMoEConfig as MoEConfig
from mlops.expert_parallel.quack.router import route_op
from torch import nn
from torch.nn import functional as F

from workloads.common import RotaryEmbedding
from workloads.mlops.common import RMSNorm
from workloads.mlops.olmoe import Attention
from workloads.mlops.olmoe import Block as BaseBlock
from workloads.mlops.olmoe import OLMoE as BaseOLMoE
from workloads.pytorch.olmoe import OLMoEConfig


def _expert_options(
    config,
    ep_group,
    router_dtype,
    compute_precision,
    weight_grad_dtype,
    activation_transport,
):
    return MoEConfig(
        ep_size=torch.distributed.get_world_size(ep_group),
        num_experts=config.n_experts,
        top_k=config.top_k,
        model_dim=config.d_model,
        expert_hidden_dim=config.d_ff_expert,
        compute_precision=compute_precision,
        weight_grad_dtype=weight_grad_dtype,
        activation_transport=activation_transport,
        router_dtype=router_dtype,
        router_weight_grad_dtype=router_dtype,
        renormalize_topk=False,
        share_expert_banks=True,
    )


class MoE(nn.Module):
    def __init__(
        self,
        config: OLMoEConfig,
        *,
        ep_group,
        buffer,
        device,
        parameter_device,
        router_dtype,
        compute_precision,
        weight_grad_dtype,
        activation_transport,
    ):
        super().__init__()
        self.config = config
        options = _expert_options(
            config,
            ep_group,
            router_dtype,
            compute_precision,
            weight_grad_dtype,
            activation_transport,
        )
        self.experts = QuackMoE(options, ep_group, buffer=buffer, device=device)
        if parameter_device != device:
            # Copy the logical parameters, not their larger communication-bank
            # storage. The layer publishes relocated values before computing.
            for name, parameter in tuple(self.experts.named_parameters()):
                value = parameter.detach().to(parameter_device, copy=True).contiguous()
                self.experts.register_parameter(name, nn.Parameter(value))

    def forward(self, hidden, residual, *, return_metrics=False):
        flat = hidden.reshape(-1, self.config.d_model)
        router = self.experts.router_weight
        logits = F.linear(flat.to(router.dtype), router).float()
        weights, ids, counts = route_op(logits, self.config.top_k, False)
        routed = self.experts(flat, expert_ids=ids, routing_weights=weights)
        probability_sum = logits.softmax(-1).sum(0)
        frequency = counts.float() / (flat.shape[0] * self.config.top_k)
        auxiliary = (
            self.config.n_experts * (frequency * probability_sum / flat.shape[0]).sum()
        )
        output = residual + routed.reshape_as(hidden)
        if return_metrics:
            return output, auxiliary, counts, probability_sum
        return output, auxiliary


class Block(BaseBlock):
    def __init__(
        self,
        config: OLMoEConfig,
        *,
        ep_group,
        buffer,
        device,
        parameter_device,
        router_dtype,
        compute_precision,
        weight_grad_dtype,
        activation_transport,
    ):
        nn.Module.__init__(self)
        self.attn_norm = RMSNorm(config.d_model).to(dtype=torch.bfloat16)
        self.attn = Attention(config).to(dtype=torch.bfloat16)
        self.ffn_norm = RMSNorm(config.d_model).to(dtype=torch.bfloat16)
        self.moe = MoE(
            config,
            ep_group=ep_group,
            buffer=buffer,
            device=device,
            parameter_device=parameter_device,
            router_dtype=router_dtype,
            compute_precision=compute_precision,
            weight_grad_dtype=weight_grad_dtype,
            activation_transport=activation_transport,
        )


class OLMoE(BaseOLMoE):
    """MLOps BF16 attention/head and QuackMoE BF16 or FP8 routed experts.

    ``token_capacity`` creates one model-owned MoonEP buffer shared by all blocks.
    Alternatively, ``buffers`` supplies borrowed resources, one entry per block.
    Expert publication/reduction banks are shared along with each token buffer;
    model parameters remain distinct. ``expert_parameters`` identifies unique
    home shards; all other parameters are replicas within this EP group.
    Parameters normally start on the compute device. ``parameter_device="cpu"``
    also supports callers that materialize compute values from host state.
    Experts default to BF16; ``compute_precision="fp8_current"`` selects FP8.
    Expert gradient precision and activation transport are independent options.
    The router supports BF16 or FP32 (the default).
    """

    def __init__(
        self,
        config: OLMoEConfig,
        *,
        ep_group,
        token_capacity: int | None = None,
        buffers: Sequence | None = None,
        device=None,
        parameter_device=None,
        router_dtype=torch.float32,
        compute_precision="bf16",
        weight_grad_dtype=torch.bfloat16,
        activation_transport="bf16",
    ):
        nn.Module.__init__(self)
        if (buffers is None) == (token_capacity is None):
            raise ValueError("Supply token_capacity or existing buffers, exclusively")
        if buffers is not None and len(buffers) != config.n_layers:
            raise ValueError(
                "Supply one buffer entry per block; entries may share a buffer"
            )
        if token_capacity is not None and (
            type(token_capacity) is not int or token_capacity <= 0
        ):
            raise ValueError("token_capacity must be a positive integer")
        device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        if device.type != "cuda":
            raise ValueError("QuackMoE computation requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        parameter_device = (
            torch.device(parameter_device) if parameter_device is not None else device
        )
        if parameter_device != device and parameter_device.type != "cpu":
            raise ValueError("Parameters must start on the compute device or CPU")
        self.config = config
        # Runtime handles retain the resources. Model copies used for capture
        # must not attempt to copy CUDA streams or device mappings.
        self._owns_buffer = False
        owned_buffer = None
        self.blocks = nn.ModuleList()
        if buffers is None:
            from mlops.expert_parallel import create_buffer

            owned_buffer = create_buffer(
                _expert_options(
                    config,
                    ep_group,
                    router_dtype,
                    compute_precision,
                    weight_grad_dtype,
                    activation_transport,
                ),
                token_capacity,
                ep_group,
            )
            buffers = [owned_buffer] * config.n_layers
        try:
            with torch.device(parameter_device):
                self.embed = nn.Embedding(
                    config.vocab_size, config.d_model, dtype=torch.bfloat16
                )
                self.rotary = RotaryEmbedding(
                    config.head_dim, base=config.rope_base, capacity=config.max_seq_len
                )
                for buffer in buffers:
                    self.blocks.append(
                        Block(
                            config,
                            ep_group=ep_group,
                            buffer=buffer,
                            device=device,
                            parameter_device=parameter_device,
                            router_dtype=router_dtype,
                            compute_precision=compute_precision,
                            weight_grad_dtype=weight_grad_dtype,
                            activation_transport=activation_transport,
                        )
                    )
                self.final_norm = RMSNorm(config.d_model).to(dtype=torch.bfloat16)
                self.lm_head = nn.Linear(
                    config.d_model, config.vocab_size, bias=False, dtype=torch.bfloat16
                )
        except BaseException:
            self.close()
            if owned_buffer is not None:
                owned_buffer.destroy()
            raise
        self._owns_buffer = owned_buffer is not None

    def expert_parameters(self):
        for block in self.blocks:
            yield from block.moe.experts.expert_parameters()

    def close(self):
        buffer = (
            self.blocks[0].moe.experts.communication_buffer
            if self._owns_buffer
            else None
        )
        self._owns_buffer = False
        for block in self.blocks:
            block.moe.experts.close()
        if buffer is not None:
            buffer.destroy()


__all__ = ["OLMoE", "OLMoEConfig"]
