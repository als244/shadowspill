"""Main attention/feed-forward projections and routed expert banks."""

TARGETS = (
    "blocks.*.attn.wq",
    "blocks.*.attn.wk",
    "blocks.*.attn.wv",
    "blocks.*.attn.wo",
    "blocks.*.moe",
)

# Main shared-expert projections; scalar mixing gates stay frozen.
SHARED_TARGETS = ()
