"""Host observations: global token mean, held-out evaluation and a learning check."""

import json
import math

import torch
import torch.distributed as dist

from workloads.recipes.text.glm import fixed_batch_learning_report

from .inputs import Inputs


def global_loss(local):
    # Losses are already divided by the global token count. Sum rank
    # contributions; averaging here would normalize a second time.
    total = torch.tensor(local, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(total)
    result = total.item()
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite global loss")
    return result


def training_metrics(observations):
    return {"global_mean_loss": global_loss(sum(observations.losses))}


def evaluate(trainer, options, *, rank, stream):
    source = Inputs(options, split="val", rank=rank, world_size=options["ranks"])
    total = 0.0
    for _ in range(options["eval_updates"]):
        calls = next(source)
        for args, kwargs in zip(calls["args"], calls["kwargs"], strict=True):
            total += float(trainer.evaluate(args=[args], kwargs=[kwargs]))
    record = {
        "step": trainer.step_count,
        "validation_loss": global_loss(total / options["eval_updates"]),
    }
    if rank == 0:
        stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def learning_report(root, options, losses, rank):
    threshold = options["minimum_relative_loss_drop"]
    if options["repeat_training_batch"] and len(losses) >= 4:
        report = fixed_batch_learning_report(
            losses, minimum_relative_drop=threshold or 0
        )
        if rank == 0:
            (root / "learning.json").write_text(json.dumps(report, indent=2) + "\n")
            print("fixed-batch learning: " + json.dumps(report), flush=True)
        if threshold is not None and not report["passed"]:
            raise RuntimeError(
                "fixed-batch loss did not meet the requested decrease; "
                "see learning.json"
            )
    elif threshold is not None:
        raise ValueError("loss-drop check needs four newly executed updates")
