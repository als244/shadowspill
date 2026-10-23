"""Main attention/feed-forward projections and routed expert banks.

KDA control/output gates and indexers are excluded."""

TARGETS = (
    "layers.*.self_attn.q_a_proj",
    "layers.*.self_attn.q_b_proj",
    "layers.*.self_attn.kv_a_proj_with_mqa",
    "layers.*.self_attn.kv_b_proj",
    "layers.*.self_attn.o_proj",
    "layers.*.self_attn.q_proj",
    "layers.*.self_attn.k_proj",
    "layers.*.self_attn.v_proj",
    "layers.*.mlp.gate_proj",
    "layers.*.mlp.up_proj",
    "layers.*.mlp.down_proj",
    "layers.*.mlp",
    "visual.*.qkv",
    "visual.*.proj",
    "visual.*.gate_proj",
    "visual.*.up_proj",
    "visual.*.down_proj",
)

# Main shared-expert projections; scalar mixing gates stay frozen.
SHARED_TARGETS = (
    "layers.*.mlp.shared_experts.gate_proj",
    "layers.*.mlp.shared_experts.up_proj",
    "layers.*.mlp.shared_experts.down_proj",
)
