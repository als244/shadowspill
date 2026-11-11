"""Persist recipe choices and reject inconsistent phase/resume settings."""

from fractions import Fraction

from workloads.recipes.text.phases.candidates import batch_sizes


def build_variants(options):
    from workloads.recipes.text.phases.variants import build_variants as variants

    return variants(options, alignment=32)


DEFAULTS = {
    "steps": 5,
    "execution_gib": 76,
    "spill_gib": 96,
}


def configure(args):
    """CLI test/helper spelling of the centralized configuration contract."""
    from workloads.recipes.training.configuration import configure as shared

    values = vars(args).copy()
    phase = values.pop("phase")
    values = {key: value for key, value in values.items() if value is not None}
    if phase not in {"all", "build"}:
        values = {
            key: value
            for key, value in values.items()
            if key
            in {
                "run_dir",
                *DEFAULTS,
                "resume",
                "microbatch_sizes",
                "nsys_steps",
                "nsys_start_step",
            }
        }
    return shared("workloads.recipes.text.glm53_flash_lora.adapter", values, phase)


def validate(options):
    positive = (
        "sequence_length",
        "sequences_per_step",
        "steps",
        "lora_rank",
        "head_chunk_size",
        "execution_gib",
        "spill_gib",
        "ranks",
        "eval_updates",
        "measure_seconds",
        "learning_rate",
    )
    if any(options[key] <= 0 for key in positive):
        raise ValueError(
            "shapes, steps, ranks, budgets, rates and measurement duration "
            "must be positive"
        )
    batch_sizes(options)
    if any(
        options[key] < 0
        for key in (
            "eval_every",
            "checkpoint_every",
            "external_headroom_mib",
            "warmup_seconds",
        )
    ):
        raise ValueError("cadences, headroom and warmup must be nonnegative")
    if options["schedule_steps"] < max(2, options["steps"]):
        raise ValueError("schedule-steps must exceed one and cover steps")
    threshold = options["minimum_relative_loss_drop"]
    if threshold is not None:
        if not options["repeat_training_batch"]:
            raise ValueError("a loss-drop requirement needs --repeat-training-batch")
        if not 0 <= threshold < 1 or options["steps"] < 4:
            raise ValueError("loss-drop must lie in [0, 1), with at least four steps")
    if any(not 0 <= Fraction(value) <= 1 for value in options["resolution_options"]):
        raise ValueError("recomputation fractions must lie in [0, 1]")
