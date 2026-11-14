"""GLM-specific Python defaults and validation for the shared runner."""

DEFAULTS = {
    "steps": 5,
    "execution_gib": 76,
    "spill_gib": 96,
    "checkpoint": None,
    "frozen_state_id": None,
    "lora_rank": 32,
    "lora_alpha": 32,
    "router_weight_dtype": None,
    "learning_rate": 1e-4,
    "head_chunk_size": 1024,
    "layer_limit": None,
    "external_headroom_mib": 2048,
    "resolution_options": ["1"],
    "repeat_training_batch": False,
    "minimum_relative_loss_drop": None,
    "minimum_evict_bytes": 128 << 10,
}


def validate(options):
    if (
        type(options["lora_rank"]) is not int
        or options["lora_rank"] <= 0
        or options["lora_alpha"] <= 0
        or type(options["head_chunk_size"]) is not int
        or options["head_chunk_size"] <= 0
    ):
        raise ValueError("LoRA rank, alpha and head chunk size must be positive")
    threshold = options["minimum_relative_loss_drop"]
    if threshold is not None:
        if not options["repeat_training_batch"]:
            raise ValueError("a loss-drop requirement needs --repeat-training-batch")
        if not 0 <= threshold < 1 or options["steps"] < 4:
            raise ValueError("loss-drop must lie in [0, 1), with at least four steps")
