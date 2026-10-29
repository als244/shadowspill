"""Bounded base loading and chunk-independent initialization of LoRA and RoPE."""

from workloads.recipes.text.initialization import FromCheckpoint


class Initialize(FromCheckpoint):
    """The generic checkpoint initializer at this recipe's options."""

    def __init__(self, options):
        super().__init__(
            "qwen35moe",
            options["checkpoint"],
            seed=options["seed"],
            max_seq_len=options["sequence_length"],
        )
