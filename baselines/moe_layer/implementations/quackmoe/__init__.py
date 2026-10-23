"""QuackMoE with its native MoonEP communication buffer."""

import torch

from ..api import ModuleImplementation
from ..tensors import expert_tensor, tensor


def build(
    shape, *, precision, training, tokens, world_size, seed, router_weight_dtype="bf16"
):
    from mlops.expert_parallel import (
        LoRAConfig,
        QuackMoE,
        QuackMoEConfig,
        QuackMoELoRA,
    )
    from mlops.expert_parallel.buffers import create_buffer

    config = QuackMoEConfig(
        ep_size=world_size,
        num_experts=shape.experts,
        top_k=shape.top_k,
        model_dim=shape.width,
        expert_hidden_dim=shape.hidden,
        compute_precision=precision,
        activation_transport="bf16",
        weight_grad_dtype=torch.float32,
        router_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[
            router_weight_dtype
        ],
        router_weight_grad_dtype=torch.float32,
        num_comm_sms=32,
        profile_ranges=False,
    )
    buffer = create_buffer(config, tokens, torch.distributed.group.WORLD)
    options = dict(buffer=buffer, device=torch.device("cuda", 0))
    first_expert = torch.distributed.get_rank() * (shape.experts // world_size)
    if training == "lora":
        layer = QuackMoELoRA(
            config,
            lora=LoRAConfig(rank=shape.lora_rank, alpha=shape.lora_alpha),
            **options,
        )
        with torch.no_grad():
            for index, parameter in enumerate(layer.lora_parameters()):
                parameter.copy_(
                    expert_tensor(
                        tuple(parameter.shape),
                        seed=seed,
                        slot=2 + index,
                        first_expert=first_expert,
                        std=0.015,
                    )
                )
    else:
        layer = QuackMoE(config, **options)
    with torch.no_grad():
        for slot, parameter in enumerate(layer.expert_parameters()):
            parameter.copy_(
                expert_tensor(
                    tuple(parameter.shape),
                    seed=seed,
                    slot=slot,
                    first_expert=first_expert,
                )
            )
    # Same router and input distribution on both implementations; real top-k.
    with torch.no_grad():
        layer.router_weight.copy_(
            tensor((shape.experts, shape.width), seed=seed, dtype=config.router_dtype)
        )
    return ModuleImplementation(
        layer,
        cleanup=(buffer.destroy,),
        metadata={
            "expert_backend": "QuackMoE",
            "dispatcher": "MoonEP",
            "gradient_dtype": "float32",
            "router_weight_dtype": router_weight_dtype,
            "router_logit_dtype": "float32",
        },
    )
