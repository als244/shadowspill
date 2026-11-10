"""Explicit workload choices using generic compiler and provider contracts."""

from mlops.integrations.shadowspill import sonicmoe_chunk_rules

from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule
from workloads.models.shared.chunking import operator_rules
from workloads.models.shared.execution import text_execution_chunks


def build_options(options, batch_size):
    size = options.get("chunk_size", 0)
    policy = options.get("graph_pair_policy", "cost-input")
    retained = (
        RetentionRule("mlops.flash_attention_builtin_aten_fwd.default", "retain"),
    )
    local = (
        tuple(
            RetentionRule(target, "recompute")
            for target in (
                "mlops.sonicmoe_routing_metadata.default",
                "mlops.lora_sonicmoe_up_fwd.default",
                "mlops.lora_sonicmoe_down_fwd.default",
                "mlops.lora_assignment_layout.default",
                "mlops.lora_dispatch.default",
                "mlops.lora_grouped_linear_fwd.default",
                "mlops.swiglu_builtin_triton_fwd.default",
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
    if size:
        result["chunking"] = text_execution_chunks(
            sequence_length=options["sequence_length"],
            batch_size=batch_size,
            chunk_size=size,
            expert_chunk_size=options.get("moe_chunk_size", 0) or size,
            rules=(*operator_rules(batch_size), *sonicmoe_chunk_rules()),
            expert_targets=(
                *(
                    f"mlops.lora_sonicmoe_{direction}_{phase}.default"
                    for direction in ("up", "down")
                    for phase in ("fwd", "bwd")
                ),
                "mlops.lora_grouped_linear_fwd.default",
                "mlops.lora_grouped_linear_bwd.default",
            ),
            inner_targets=("mlops.lora_router_logits.default",),
            variants=("save", "recompute"),
        )
    return result
