"""Main attention/feed-forward projections and routed expert banks.

QKV+Z stays packed; beta/decay controls are excluded."""

TARGETS = (
    "blocks.*.mixer.wq",
    "blocks.*.mixer.wk",
    "blocks.*.mixer.wv",
    "blocks.*.mixer.wo",
    "blocks.*.mixer.w_qkvz",
    "blocks.*.mixer.w_out",
    "blocks.*.moe",
)

# Main shared-expert projections; scalar mixing gates stay frozen.
SHARED_TARGETS = ("blocks.*.moe.shared.gate_up", "blocks.*.moe.shared.down")
