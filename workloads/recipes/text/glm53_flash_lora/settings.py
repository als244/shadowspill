"""Persist recipe choices and reject inconsistent phase/resume settings."""

import json
from fractions import Fraction
from pathlib import Path

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
        for key, default in DEFAULTS.items():
            if options[key] is None:
                options[key] = default
        for key in ("data", "checkpoint"):
            options[key] = str(Path(options[key]).expanduser().resolve())
        options["minimum_evict_bytes"] = 128 << 10
    validate(options)
    if args.phase == "communicate" and options["ranks"] == 1:
        raise ValueError("single-GPU profiling already measures host communication")
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(options, indent=2) + "\n")
    return root, options


def validate(options):
    positive = (
        "sequence_length",
        "sequences_per_step",
        "max_sequences_per_example",
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
