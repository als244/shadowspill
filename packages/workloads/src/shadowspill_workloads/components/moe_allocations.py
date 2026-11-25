"""Exact fixed-geometry contracts for the supplied MoE MLOps/FLA paths.

Dropless routed experts allocate T*K assignment rows, E+1 offsets and fixed
expert banks. Sorting changes row order only. Grouped GEMMs allocate full row
outputs and E+1 tile-prefix entries; the device-side counts change execution,
never allocation. LoRA uses those same bounded operators and fixed factor sizes.
The trace-visible MoE provider splits the same operations into prepare/finish
regions, with the same assignment, expert-bank and tile-prefix extents.

These guarantees pin provider Python and extension code through registration.
They do not replace profiling or joint admission, or cover a different provider.
In particular, SonicMoE/EP paths require their own contracts and qualification.
"""

from .hybrid_allocations import CONTRACTS as HYBRID_CONTRACTS
from .text_allocations import CONTRACTS as TEXT_CONTRACTS

CONTRACT_ID = "shadowspill/moe-fixed-provider-allocation@1"

CONTRACTS = {
    **TEXT_CONTRACTS,
    **HYBRID_CONTRACTS,
    **{
        f"torch.ops.mlops.{name}_{direction}.default": CONTRACT_ID
        for name in (
            "moe_builtin_grouped_gemm",
            "moe_builtin_grouped_gemm_prepare",
            "moe_builtin_grouped_gemm_finish",
        )
        for direction in ("fwd", "bwd")
    },
    **{
        f"torch.ops.mlops.{name}.default": CONTRACT_ID
        for name in (
            "lora_router_logits",
            "lora_router_logits_bwd",
            "lora_route",
            "lora_route_bwd",
            "lora_assignment_layout",
            "lora_dispatch",
            "lora_dispatch_bwd",
            "lora_combine",
            "lora_sum_rows",
            "lora_combine_bwd",
            "lora_grouped_linear_fwd",
            "lora_grouped_linear_bwd",
        )
    },
}
