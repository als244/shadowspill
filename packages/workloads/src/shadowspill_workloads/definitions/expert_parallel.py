"""Portable supplied expert resources; provider objects belong to workers."""

from contextlib import contextmanager
from dataclasses import asdict, replace
from math import isfinite

from shadowspill import Parallelism, ParameterStorage, Replicated, Resource, Sharded
from shadowspill.parallelism.selectors import select
from shadowspill.training.definitions import ComponentLayout, ModelBuild
from shadowspill_client._records import plain

from .expert_allocations import CONTRACTS as EXPERT_CONTRACTS
from .moe_allocations import CONTRACTS


def describe(config, family, parallelism, tokens, *, grad_dtype):
    """Declare the existing shared Quack buffer without Torch or a live group.

    Only explicit workload choices are serialized. The worker constructs the
    provider's own configuration, retaining its defaults and ABI validation.
    Tokens name the exact physical input shape, including scheduled padding.
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


def validate(config):
    if config.implementation == "mlops_ep":
        if config.compute_dtype != "bfloat16":
            raise ValueError("Quack EP requires BF16 compute")
        if config.lora is not None:
            raise ValueError("the installed EP definitions support full training")
    elif config.expert_options is not None:
        raise ValueError("expert resource settings require the mlops_ep implementation")


def validate_ownership(architecture, parallelism):
    ranks = parallelism.ranks_per_replica
    if parallelism.groups.get("all_ranks") != (tuple(range(ranks)),):
        raise ValueError("Quack EP requires one ordered all_ranks group per replica")
    if parallelism.default_parameter_placement != Replicated("all_ranks"):
        raise ValueError("Quack EP requires replicated dense model parameters")
    for layer in range(architecture.n_layers):
        for projection in ("gate_up_weight", "down_weight"):
            name = f"blocks.{layer}.moe.experts.{projection}"
            placement = select(
                (name,),
                parallelism.parameter_placements,
                parallelism.default_parameter_placement,
            )
            if placement != Sharded("all_ranks", 0):
                raise ValueError(f"{name}: Quack EP requires expert-axis sharding")
            storage = select(
                (name,), parallelism.parameter_storage, ParameterStorage()
            ).inherit(parallelism.default_parameter_storage)
            for component in ("weights", "gradients", "optimizer"):
                group = parallelism.group(storage.group_for(component))
                if any(len(members) != 1 for members in group.groups):
                    raise ValueError(
                        f"{name}: expert {component} storage must be local"
                    )


def build(config, architecture, family, context):
    """Bind each exact physical size to the existing named resource lifetime.

    Quack validates exact input rows, not a maximum capacity. Different prepared
    sizes therefore have distinct handles/resources, included in joint admission;
    repeated operations at the same size reuse the same resource description.
    Parameter names and logical ownership are independent of those handles.
    """
    validate_ownership(architecture, context.parallelism)
    (tokens,) = context.layout.local_shape(("inputs", "tokens"))
    resource = describe(
        architecture, family, context.parallelism, tokens, grad_dtype=config.grad_dtype
    )
    name = f"experts.tokens-{tokens}"
    options = plain(resource.options)
    options["layers"] = [f"{name}.{layer}" for layer in options["layers"]]
    resource = replace(resource, options=options)
    return ModelBuild(
        {**asdict(config), "expert_options": options},
        resources={name: resource},
        grad_dtype=config.grad_dtype,
        # No Sonic preparation is attached to Quack EP. Its resource owns fixed
        # GEMM choices; the exact BF16 calls have separate allocation contracts.
        allocation_invariants={**CONTRACTS, **EXPERT_CONTRACTS},
        component_layouts=component_layouts(architecture, family, config.grad_dtype),
    )


def component_layouts(architecture, family, grad_dtype):
    """BF16ComputeWeight stores one same-coordinate BF16 payload per parameter."""
    if grad_dtype == "bfloat16":
        return ()
    fields = {"experts.gate_up_weight": 3, "experts.down_weight": 3}
    if family != "olmoe":
        fields["experts.router_weight"] = 2
    return tuple(
        ComponentLayout(
            f"blocks.{layer}.moe.{field}",
            ("_data",),
            tuple(range(rank)),
            "mlops/bf16-compute@1",
        )
        for layer in range(architecture.n_layers)
        for field, rank in fields.items()
    )


def initialization(config, architecture, family, context):
    from shadowspill_workloads.initialization.pretrained import describe as base

    validate_ownership(architecture, context.parallelism)
    if context.temporary is not None:
        return base(family, architecture, context)
    if context.weights.kind != "random":
        raise ValueError(
            "installed EP initialization requires random weights or resume"
        )
    description = base(family, architecture, context)

    @contextmanager
    def open_experts(runtime):
        import torch

        from shadowspill_workloads.initialization.expert_parallel import Initialize

        with description.open(runtime) as catalog:
            yield Initialize(
                catalog,
                expert_rank=context.logical_rank
                % context.parallelism.ranks_per_replica,
                std=architecture.initializer_range,
                router_dtype=torch.float32 if family == "olmoe" else torch.bfloat16,
            )

    return replace(description, open=open_experts)
