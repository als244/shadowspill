"""GLM geometry bound to the shared Quack LoRA resource and generic ownership."""

from dataclasses import asdict

from shadowspill import Resource
from shadowspill.training.definitions import ModelBuild

from ..expert_parallel import validate_parameters
from .components import layers, layouts

PARAMETERS = (
    "gate_up_weight",
    "down_weight",
    "lora_gate_up_a",
    "lora_gate_up_b",
    "lora_down_a",
    "lora_down_b",
)


def validate(config):
    if config.lora is None:
        raise ValueError("GLM expert parallelism requires LoRA")
    if config.lora_dtype != "bfloat16" or config.lora["rank"] % 16:
        raise ValueError("EP LoRA requires BF16 factors and rank divisible by 16")
    if config.architecture["expert_storage"] not in ("bf16", "fp8_block"):
        raise ValueError("EP LoRA supports BF16 or native block-FP8 expert storage")


def ownership(config, parallelism):
    validate_parameters(
        parallelism,
        (
            f"layers.{layer}.mlp.experts.{name}"
            for layer in layers(config.architecture)
            for name in PARAMETERS
        ),
    )


def describe(config, parallelism, tokens):
    validate(config)
    ownership(config, parallelism)
    c = config.catalog.Config(**config.architecture)
    ranks = parallelism.ranks_per_replica
    if not 2 <= ranks <= 128 or c.num_local_experts % ranks:
        raise ValueError("expert count must divide evenly over 2..128 ranks")
    if c.hidden_size % 128 or c.moe_intermediate_size % 128:
        raise ValueError("expert feature/hidden widths must be multiples of 128")
    name = f"experts.tokens-{tokens}"
    return name, Resource(
        "shadowspill_workloads.models.shared.expert_parallel.resources:initialize",
        {
            "config": {
                "ep_size": ranks,
                "num_experts": c.num_local_experts,
                "top_k": c.num_experts_per_tok,
                "model_dim": c.hidden_size,
                "expert_hidden_dim": c.moe_intermediate_size,
                "tokens_per_rank": tokens,
                "router_dtype": "bfloat16",
                "router_weight_grad_dtype": "bfloat16",
                "weight_grad_dtype": config.grad_dtype,
                "compute_precision": c.expert_storage,
                "share_expert_banks": True,
                "init_std": c.initializer_range,
                "swiglu_limit": c.swiglu_limit,
                "gemm_tuned": False,
            },
            "lora": {
                "rank": config.lora["rank"],
                "alpha": config.lora["alpha"],
                "gradient_dtype": config.grad_dtype,
            },
            "layers": [
                f"{name}.layers.{layer}.mlp.experts"
                for layer in layers(config.architecture)
            ],
        },
        {"exchange": parallelism.group("all_ranks")},
    )


def build(config, context):
    from shadowspill_client._records import plain

    from .allocations import contracts, preparations

    (tokens,) = context.layout.local_shape(("inputs", "tokens"))
    name, resource = describe(config, context.parallelism, tokens)
    return ModelBuild(
        {**asdict(config), "expert_options": plain(resource.options)},
        resources={name: resource},
        grad_dtype=config.grad_dtype,
        operator_preparation=preparations(config),
        allocation_invariants=contracts(config),
        component_layouts=layouts(config),
    )
