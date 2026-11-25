"""Adapt main attention projections and routed expert MLPs on each physical layer."""

TARGETS = (
    "layers.*.attn.wq_du",
    "layers.*.attn.wk_dv",
    "layers.*.attn.wv_dv",
    "layers.*.attn.wo_ud",
    "layers.*.mlp",
)
