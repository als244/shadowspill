"""Text-specific defaults and hooks for the model-independent training runner."""

import math
from fractions import Fraction

from .phases.candidates import batch_sizes, build_candidates
from .phases.variants import build_variants

DEFAULTS = dict(
    data=None,
    sequence_length=32768,
    sequences_per_step=1,
    microbatch_sizes=[1],
    chunk_sizes=[0],
    moe_chunk_sizes=[0],
    graph_pair_policies=["cost-input"],
    schedule_steps=None,
    learning_rate=2e-4,
    seed=0,
    initialization="checkpoint",
)


def validate(options):
    if options["initialization"] not in {"checkpoint", "random"}:
        raise ValueError("initialization must be checkpoint or random")
    if not options["data"]:
        raise ValueError("data is required")
    if options["schedule_steps"] is None:
        options["schedule_steps"] = max(2, options["steps"])
    for name in ("sequence_length", "schedule_steps"):
        if type(options[name]) is not int or options[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if options["schedule_steps"] < max(2, options["steps"]):
        raise ValueError("schedule_steps must exceed one and cover steps")
    rate = options["learning_rate"]
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int | float)
        or not math.isfinite(rate)
        or rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    options["microbatch_sizes"] = list(batch_sizes(options))
    if any(
        not 0 <= Fraction(value) <= 1 for value in (options["resolution_options"] or ())
    ):
        raise ValueError("recomputation fractions must lie in [0, 1]")


def frozen_state_identity(options, metadata):
    """Keep generated bases distinct from pretrained checkpoint identities."""
    if options["initialization"] == "checkpoint":
        return options["frozen_state_id"]
    import hashlib
    import json

    value = json.dumps(
        {
            "initializer": "bounded-random-v1",
            "seed": options["seed"],
            "model": metadata,
        },
        sort_keys=True,
    ).encode()
    return "random-v1:" + hashlib.sha256(value).hexdigest()


def candidates(options, *, alignment=1):
    variants = build_variants(options, alignment=alignment)
    configured = dict(options)
    if (
        len(variants) != 1
        or variants[0]["chunk_size"]
        or variants[0]["graph_pair_policy"] != "cost-input"
    ):
        configured["build_variants"] = variants
    return tuple(
        {
            **item,
            "num_examples": options["sequences_per_step"]
            // item["sequences_per_example"],
        }
        for item in build_candidates(configured)
    )


def check_source(source, options, split):
    count = options["steps"] if split == "train" else options["eval_updates"]
    if len(source.data) < count:
        raise ValueError(
            f"{split} corpus does not contain {count} complete global updates"
        )
    return source


def global_loss(local):
    import math

    import torch
    import torch.distributed as dist

    value = torch.tensor(local, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(value)
    result = value.item()
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite global loss")
    return result


def training_metrics(observations):
    # Objectives normalize by global tokens. Reduce their contributions once.
    return {"global_mean_loss": global_loss(sum(observations.losses))}


def trainer_options(options, metadata):
    from shadowspill.training.schedules import WarmupCosine

    return dict(
        schedules={
            "optimizer_group.0.lr": WarmupCosine(
                lr=options["learning_rate"],
                min_lr=options["learning_rate"] / 10,
                warmup_steps=1,
                total_steps=options["schedule_steps"],
            )
        },
        metric_reducer=training_metrics,
        frozen_state_id=metadata.get("frozen_state_id"),
    )
