"""Explicit provider contracts for GLM's ordinary packed-token recipe."""

from mlops.integrations.shadowspill import mhc_chunk_rules, sonicmoe_chunk_rules

from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule
from workloads.models.shared.chunking import operator_rules
from workloads.models.shared.execution import text_execution_chunks

EXPERT_TARGETS = tuple(
    f"mlops.lora_sonicmoe_{direction}_{phase}.default"
    for direction in ("up", "down")
    for phase in ("fwd", "bwd")
)
ATTENTION_TARGETS = (
    "mlops.kimi_delta_attention_cula_sm90_native_fwd.default",
    "mlops.kimi_delta_attention_cula_sm90_fwd.default",
    "mlops.kimi_delta_attention_fla_fwd.default",
    "mlops.sparse_latent_attention_hopper_fwd.default",
    "mlops.sparse_latent_attention_tilelang_fwd.default",
)


def build_options(options, batch_size):
    size = options.get("chunk_size", 0)
    policy = options.get("graph_pair_policy", "cost-input")
    if policy not in {"cost-input", "cost-attention", "attention-input"}:
        raise ValueError("unknown GLM graph-pair policy")
    retained = tuple(RetentionRule(target, "retain") for target in ATTENTION_TARGETS)
    local = (
        tuple(
            RetentionRule(target, "recompute")
            for target in (
                "mlops.sonicmoe_routing_metadata.default",
                "mlops.lora_sonicmoe_up_fwd.default",
                "mlops.lora_sonicmoe_down_fwd.default",
            )
        )
        if size
        else ()
    )
    pair = GraphPairPolicy(
        save=RetentionPolicy(default="recompute", rules=(*retained, *local))
        if policy == "attention-input"
        else RetentionPolicy(rules=local),
        recompute=RetentionPolicy(default="recompute", rules=retained)
        if policy == "cost-attention"
        else RetentionPolicy(default="recompute"),
    )
    result = {"graph_pair_policy": pair}
    if not size:
        return result
    expert_size = options.get("moe_chunk_size", 0) or size
    if min(size, expert_size) <= 0 or any(
        v % 32 for v in (size, expert_size, options["sequence_length"])
    ):
        raise ValueError(
            "GLM fused mHC execution requires sequence and chunk sizes divisible by 32"
        )
    rules = (
        *operator_rules(batch_size),
        *mhc_chunk_rules(batch_size=batch_size),
        *sonicmoe_chunk_rules(compute_precision="fp8_block"),
    )
    result["chunking"] = text_execution_chunks(
        sequence_length=options["sequence_length"],
        batch_size=batch_size,
        chunk_size=size,
        expert_chunk_size=expert_size,
        rules=rules,
        expert_targets=EXPERT_TARGETS,
        variants=("save", "recompute"),
        inner_targets=(
            "mlops.mhc_pre_deepgemm.default",
            "mlops.mhc_post_tilekernels.default",
            "mlops.lora_router_logits.default",
        ),
    )
    return result
