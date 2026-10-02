"""OLMoE decoder with MLOps attention and QuackMoE expert parallelism.

The caller supplies the process group and MoonEP buffers. This module knows
nothing about planning or training backends; it is an ordinary PyTorch model
with fixed-device communication resources, like its QuackMoE submodules.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from quack_moe import MoEConfig, QuackMoE
from quack_moe.router import route_op
from torch import nn
from torch.nn import functional as F

from workloads.common import RotaryEmbedding
from workloads.mlops.common import RMSNorm
from workloads.mlops.olmoe import Attention
from workloads.mlops.olmoe import Block as BaseBlock
from workloads.mlops.olmoe import OLMoE as BaseOLMoE
from workloads.pytorch.olmoe import OLMoEConfig


class MoE(nn.Module):
    def __init__(
        self, config: OLMoEConfig, *, ep_group, buffer, device, parameter_device
    ):
        super().__init__()
        self.config = config
        options = MoEConfig(
            ep_size=torch.distributed.get_world_size(ep_group),
            num_experts=config.n_experts,
            top_k=config.top_k,
            model_dim=config.d_model,
            expert_hidden_dim=config.d_ff_expert,
            compute_precision="bf16",
            weight_grad_dtype=torch.bfloat16,
            renormalize_topk=False,
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
        logits = F.linear(flat.float(), self.experts.router_weight)
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
        self, config: OLMoEConfig, *, ep_group, buffer, device, parameter_device
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
        )


class OLMoE(BaseOLMoE):
    """MLOps attention/head and QuackMoE routed experts, all with BF16 compute.

    ``buffers`` supplies one caller-owned resource per block. The model's token
    capacity follows those resources. ``expert_parameters`` identifies unique
    home shards; all other parameters are replicas within this EP group.
    Parameters normally start on the compute device. ``parameter_device="cpu"``
    also supports callers that materialize compute values from host state.
    This initial workload uses BF16 experts and an FP32 router.
    """

    def __init__(
        self,
        config: OLMoEConfig,
        *,
        ep_group,
        buffers: Sequence,
        device=None,
        parameter_device=None,
    ):
        nn.Module.__init__(self)
        if len(buffers) != config.n_layers:
            raise ValueError("Supply one MoonEP buffer per transformer block")
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
        with torch.device(parameter_device):
            self.embed = nn.Embedding(
                config.vocab_size, config.d_model, dtype=torch.bfloat16
            )
            self.rotary = RotaryEmbedding(
                config.head_dim, base=config.rope_base, capacity=config.max_seq_len
            )
            self.blocks = nn.ModuleList()
            try:
                for buffer in buffers:
                    self.blocks.append(
                        Block(
                            config,
                            ep_group=ep_group,
                            buffer=buffer,
                            device=device,
                            parameter_device=parameter_device,
                        )
                    )
            except BaseException:
                self.close()
                raise
            self.final_norm = RMSNorm(config.d_model).to(dtype=torch.bfloat16)
            self.lm_head = nn.Linear(
                config.d_model, config.vocab_size, bias=False, dtype=torch.bfloat16
            )

    def expert_parameters(self):
        for block in self.blocks:
            yield from block.moe.experts.expert_parameters()

    def close(self):
        for block in self.blocks:
            block.moe.experts.close()


__all__ = ["OLMoE", "OLMoEConfig"]
