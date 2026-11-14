"""Main attention/feed-forward projections and routed expert banks."""

TARGETS = (
    "blocks.*.attn.wq",
    "blocks.*.attn.wk",
    "blocks.*.attn.wv",
    "blocks.*.attn.wo",
    "blocks.*.mlp.w1",
    "blocks.*.mlp.w3",
    "blocks.*.mlp.w2",
)

# Main shared-expert projections; scalar mixing gates stay frozen.
SHARED_TARGETS = ()
