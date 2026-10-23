"""Main attention/feed-forward projections and routed expert banks.

QKV+Z stays packed; beta/decay controls are excluded."""

TARGETS = (
    "blocks.*.mixer.wq",
    "blocks.*.mixer.wk",
    "blocks.*.mixer.wv",
    "blocks.*.mixer.wo",
    "blocks.*.mixer.w_qkvz",
    "blocks.*.mixer.w_out",
    "blocks.*.mlp.w1",
    "blocks.*.mlp.w3",
    "blocks.*.mlp.w2",
)

# Main shared-expert projections; scalar mixing gates stay frozen.
SHARED_TARGETS = ()
