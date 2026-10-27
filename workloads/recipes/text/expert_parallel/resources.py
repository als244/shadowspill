"""Declare Quack layers and their shared buffer without opening a communicator."""

import torch

from shadowspill import Parallelism, ParameterStorage, Resource, Sharded


def ownership(ranks, num_replicas):
    experts = (
        "blocks.*.moe.experts.gate_up_weight",
        "blocks.*.moe.experts.down_weight",
    )
    return Parallelism(
        ranks_per_replica=ranks,
        num_replicas=num_replicas,
        groups={
            "all_ranks": (tuple(range(ranks)),),
            "local": tuple((rank,) for rank in range(ranks)),
        },
        default_parameter_storage=ParameterStorage(
            group="all_ranks",
            weights="replicated",
            gradients="replicated",
            optimizer="sharded",
        ),
        parameter_placements={name: Sharded("all_ranks", 0) for name in experts},
        parameter_storage={name: ParameterStorage(group="local") for name in experts},
    )


def description(config, family, precision, parallelism, tokens):
    from mlops.expert_parallel import QuackMoEConfig
    from mlops.expert_parallel.quack.config import config_signature

    options = QuackMoEConfig(
        ep_size=parallelism.ranks_per_replica,
        num_experts=config.n_experts,
        top_k=config.top_k,
        model_dim=config.d_model,
        expert_hidden_dim=config.d_ff_expert,
        tokens_per_rank=tokens,
        router_dtype=torch.float32 if family == "olmoe" else torch.bfloat16,
        router_weight_grad_dtype=torch.float32
        if family == "olmoe"
        else precision.gradients,
        weight_grad_dtype=precision.gradients,
        renormalize_topk=family != "olmoe",
        share_expert_banks=True,
        init_std=getattr(config, "initializer_range", 0.02),
        gemm_tuned=False,
    )
    layers = [
        {"name": f"blocks.{index}.moe.experts", "config": config_signature(options)}
        for index in range(config.n_layers)
    ]
    resource = Resource(
        "mlops.expert_parallel.quack.resources:initialize",
        {"layers": layers},
        {"exchange": parallelism.group("all_ranks")},
    )
    return options, resource


def model(entry, config, options, precision):
    from mlops.expert_parallel import QuackMoE

    precision.apply()
    names = iter(f"blocks.{index}.moe.experts" for index in range(config.n_layers))
    return entry.model_class("mlops")(
        config,
        parameter_device="meta",
        dtype=torch.bfloat16,
        expert_factory=lambda: QuackMoE.from_resource(options, next(names)),
    )
