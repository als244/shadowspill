"""Held-out text evaluation with the same windows at each reporting step."""

import json

from workloads.recipes.text.data.calls import Inputs


def evaluate(trainer, options, stream, logger=None):
    source = Inputs(options, split="val")
    total = 0.0
    for _ in range(2):
        calls = next(source)
        for args, kwargs in zip(calls["args"], calls["kwargs"], strict=True):
            total += float(trainer.evaluate(args=[args], kwargs=[kwargs]))
    record = {"step": trainer.step_count, "validation_loss": total / 2}
    stream.write(json.dumps(record) + "\n")
    print(record, flush=True)
    if logger is not None:
        logger(record)
