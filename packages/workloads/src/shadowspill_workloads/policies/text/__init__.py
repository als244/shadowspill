"""Explicit workload choices using generic compiler and provider contracts."""

from shadowspill_workloads.integrations.mlops import (
    operator_chunk_rules,
    sonicmoe_chunk_rules,
)
from shadowspill_workloads.policies.text.execution import text_execution_chunks
from shadowspill_workloads.policies.text.retention import graph_pair_policy


def retention(name, *, chunked):
    return graph_pair_policy(
        name,
        retained_targets=("mlops.flash_attention_builtin_aten_fwd.default",),
        local_targets=(
            "mlops.sonicmoe_routing_metadata.default",
            "mlops.lora_sonicmoe_up_fwd.default",
            "mlops.lora_sonicmoe_down_fwd.default",
            "mlops.lora_assignment_layout.default",
            "mlops.lora_dispatch.default",
            "mlops.lora_grouped_linear_fwd.default",
            "mlops.swiglu_builtin_triton_fwd.default",
        )
        if chunked
        else (),
    )


def chunks(*, sequence_length, batch_size, chunk_size, expert_chunk_size):
    return text_execution_chunks(
        sequence_length=sequence_length,
        batch_size=batch_size,
        chunk_size=chunk_size,
        expert_chunk_size=expert_chunk_size,
        rules=(*operator_chunk_rules(batch_size), *sonicmoe_chunk_rules()),
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


def build_options(options, batch_size):
    size = options.get("chunk_size", 0)
    result = {
        "graph_pair_policy": retention(
            options.get("graph_pair_policy", "cost-input"), chunked=bool(size)
        )
    }
    if size:
        result["chunking"] = chunks(
            sequence_length=options["sequence_length"],
            batch_size=batch_size,
            chunk_size=size,
            expert_chunk_size=options.get("moe_chunk_size", 0) or size,
        )
    return result
