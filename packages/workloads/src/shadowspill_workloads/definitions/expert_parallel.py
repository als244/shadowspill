"""Portable supplied expert resources; provider objects belong to workers."""

from math import isfinite

from shadowspill import Parallelism, ParameterStorage, Resource, Sharded


def describe(config, family, parallelism, tokens, *, grad_dtype):
    """Declare the existing shared Quack buffer without Torch or a live group.

    Only explicit workload choices are serialized. The worker constructs the
    provider's own configuration, retaining its defaults and ABI validation.
    The caller chooses a capacity covering the complete prepared shape family.
    """
    if family not in {"qwen3moe", "qwen35moe", "olmoe"}:
        raise ValueError("unsupported expert-parallel model family")
    if grad_dtype not in {"float32", "bfloat16"}:
        raise ValueError("expert gradients must be float32 or bfloat16")
    for name, value in (
        ("tokens", tokens),
        ("layers", config.n_layers),
        ("experts", config.n_experts),
        ("top_k", config.top_k),
        ("model width", config.d_model),
        ("expert width", config.d_ff_expert),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"expert {name} must be a positive integer")
    ranks = parallelism.ranks_per_replica
    if not 2 <= ranks <= 128 or config.n_experts % ranks:
        raise ValueError("expert count must divide evenly over 2..128 ranks")
    if config.top_k > min(32, config.n_experts):
        raise ValueError("expert top_k must not exceed min(32, experts)")
    if config.d_model % 128 or config.d_ff_expert % 128:
        raise ValueError("expert feature/hidden widths must be multiples of 128")
    std = getattr(config, "initializer_range", 0.02)
    if type(std) not in (int, float) or not isfinite(std) or std <= 0:
        raise ValueError("expert initializer range must be finite and positive")
    settings = {
        "ep_size": ranks,
        "num_experts": config.n_experts,
        "top_k": config.top_k,
        "model_dim": config.d_model,
        "expert_hidden_dim": config.d_ff_expert,
        "tokens_per_rank": tokens,
        "router_dtype": "float32" if family == "olmoe" else "bfloat16",
        "router_weight_grad_dtype": "float32" if family == "olmoe" else grad_dtype,
        "weight_grad_dtype": grad_dtype,
        "renormalize_topk": family != "olmoe",
        "share_expert_banks": True,
        "init_std": std,
        "gemm_tuned": False,
    }
    return Resource(
        "shadowspill_workloads.models.shared.expert_parallel.resources:initialize",
        {
            "config": settings,
            "layers": [
                f"blocks.{index}.moe.experts" for index in range(config.n_layers)
            ],
        },
        {"exchange": parallelism.group("all_ranks")},
    )


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
