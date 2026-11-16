"""Fixed-geometry partial RoPE, convolution and gated DeltaNet contracts.

Partial RoPE writes input-sized outputs. FLA norm outputs/partials depend on
rows, widths and device geometry. Packed convolution and DeltaNet always receive
fixed-capacity cumulative/chunk tensors: NT is their allocated row count, not
the number of nonempty chunks. Histories, per-chunk weight partials, WY factors
and offsets therefore have fixed extents even for empty slots. MLOps disables
FLA's tensor cache so replay does not retain index allocations.

Registration pins the provider implementation. These contracts cover allocation
behavior; profiling and joint admission still supply sizes and physical placement.
"""

CONTRACT_ID = "shadowspill/moe-fixed-provider-allocation@1"

CONTRACTS = {
    **{
        f"torch.ops.mlops.{name}_{direction}.default": CONTRACT_ID
        for name in (
            "partial_rope_builtin_triton",
            "causal_conv_silu_fla",
            "l2_norm_fla",
            "gated_rms_norm_fla",
            "linear_attention_fla_gated_delta_rule",
        )
        for direction in ("fwd", "bwd")
    },
    "torch.ops.mlops.causal_conv_silu_fla_input_bwd.default": CONTRACT_ID,
}
