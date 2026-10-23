"""Main attention/feed-forward projections and routed expert banks."""

TARGETS = (
    "blocks.*.mixer.wq",
    "blocks.*.mixer.wk",
    "blocks.*.mixer.wv",
    "blocks.*.mixer.wo",
    "blocks.*.moe",
)

# Main shared-expert projections; scalar mixing gates stay frozen.
SHARED_TARGETS = ("blocks.*.moe.shared.gate_up", "blocks.*.moe.shared.down")
