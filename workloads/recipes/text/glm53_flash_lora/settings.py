"""Persist recipe choices and reject inconsistent phase/resume settings."""

import json
from fractions import Fraction
from pathlib import Path

from workloads.recipes.text.phases.candidates import batch_sizes, check_batch_sizes


def build_variants(options):
    """Persist a bounded, deduplicated candidate product without importing Torch."""
    result = {}
    for policy in options.get("graph_pair_policies", ["cost-input"]):
        if policy not in {"cost-input", "cost-attention", "attention-input"}:
            raise ValueError("unknown graph-pair policy")
        for size in options.get("chunk_sizes", [0]):
            for requested in options.get("moe_chunk_sizes", [0]):
                if (
                    type(size) is not int
                    or type(requested) is not int
                    or min(size, requested) < 0
                ):
                    raise ValueError(
                        "execution chunk sizes must be nonnegative integers"
                    )
                expert = (requested or size) if size else 0
                if size and any(
                    value % 32 for value in (size, expert, options["sequence_length"])
                ):
                    raise ValueError(
                        "fused mHC needs sequence and chunk sizes divisible by 32"
                    )
                if size and expert < size:
                    raise ValueError(
                        "MoE chunk size must be at least the projection chunk size"
                    )
                name = f"{policy}-chunk-{size}-moe-{expert}"
                result[name] = dict(
                    name=name,
                    chunk_size=size,
                    moe_chunk_size=expert,
                    graph_pair_policy=policy,
                )
    if not result:
        raise ValueError("at least one build candidate is required")
    return list(result.values())


DEFAULTS = {
    "steps": 5,
    "execution_gib": 76,
    "spill_gib": 96,
    "nsys_steps": 0,
    "nsys_start_step": 2,
}


def configure(args):
    root = Path(args.run_dir).resolve()
    path = root / "config.json"
    if path.exists():
        if args.phase in {"all", "build"}:
            raise ValueError("use a fresh run directory for a new build")
        options = json.loads(path.read_text())
        check_batch_sizes(options, args.microbatch_sizes)
        for key in (*DEFAULTS, "resume"):
            value = getattr(args, key)
            if value is not None:
                if (
                    key in {"execution_gib", "spill_gib"}
                    and value != options[key]
                    and args.phase != "plan"
                ):
                    raise ValueError("changed budgets require --phase plan")
                options[key] = value
    else:
        if args.phase not in {"all", "build"} or not all(
            (args.data, args.checkpoint, args.frozen_state_id)
        ):
            raise ValueError("start with --data, --checkpoint and --frozen-state-id")
        options = vars(args).copy()
        if options["microbatch_sizes"] is None:
            options["microbatch_sizes"] = [1]
        for key, default in DEFAULTS.items():
            if options[key] is None:
                options[key] = default
        for key in ("data", "checkpoint"):
            options[key] = str(Path(options[key]).expanduser().resolve())
        options["minimum_evict_bytes"] = 128 << 10
        variants = build_variants(options)
        # Preserve historical single-candidate artifact paths for default builds.
        if (
            len(variants) != 1
            or variants[0]["chunk_size"]
            or variants[0]["graph_pair_policy"] != "cost-input"
        ):
            options["build_variants"] = variants
    validate(options)
    options["microbatch_sizes"] = list(batch_sizes(options))
    if args.phase == "communicate" and options["ranks"] == 1:
        raise ValueError("single-GPU profiling already measures host communication")
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(options, indent=2) + "\n")
    return root, options


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
        "search_workers",
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
            "nsys_steps",
            "external_headroom_mib",
            "warmup_seconds",
        )
    ):
        raise ValueError("cadences, headroom and warmup must be nonnegative")
    if options["schedule_steps"] < max(2, options["steps"]):
        raise ValueError("schedule-steps must exceed one and cover steps")
    if options["nsys_start_step"] < 1:
        raise ValueError("nsys-start-step must be positive")
    if options["nsys_steps"]:
        first = options["nsys_start_step"]
        last = first + options["nsys_steps"] - 1
        if last > options["steps"]:
            raise ValueError("steps must cover the requested Nsight capture")
        for cadence in (options["eval_every"], options["checkpoint_every"]):
            if cadence and any(step % cadence == 0 for step in range(first, last)):
                raise ValueError(
                    "evaluation/checkpoint cadence must lie outside capture"
                )
    threshold = options["minimum_relative_loss_drop"]
    if threshold is not None:
        if not options["repeat_training_batch"]:
            raise ValueError("a loss-drop requirement needs --repeat-training-batch")
        if not 0 <= threshold < 1 or options["steps"] < 4:
            raise ValueError("loss-drop must lie in [0, 1), with at least four steps")
    if any(not 0 <= Fraction(value) <= 1 for value in options["resolution_options"]):
        raise ValueError("recomputation fractions must lie in [0, 1]")
