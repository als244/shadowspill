"""The common baseline clock surrounds only the selected Plan's training update."""

import statistics

import torch

from workloads.full_model import HEAD_LOSS_METRIC

from .microbatches import split_microbatches
from .reference import create


def measure(case, clock, trainer, geometry, data, stack):
    model, access, reference = create(case, trainer, geometry, stack)
    device_used, evidence = [], None
    clock.ready()
    for step in range(clock.steps):
        pair = data(step)
        args = [
            item
            for item, _ in split_microbatches(
                pair,
                size=geometry.tokens_per_microbatch,
                sequence_length=geometry.sequence_length,
            )
        ]
        expected = (
            None
            if reference is None
            else reference.step(
                model,
                *(value.to(case.device) for value in pair),
                geometry.tokens_per_microbatch,
            )
        )
        clock.begin(step)
        result = trainer.step(args=args)
        loss = float(sum(result.losses)) * case.world
        head = sum(
            float(metrics.get(HEAD_LOSS_METRIC, part * case.world))
            if isinstance(metrics, dict)
            else float(part) * case.world
            for part, metrics in zip(result.losses, result.metrics, strict=True)
        )
        clock.end(step, loss, head)
        free, total = torch.cuda.mem_get_info(case.device)
        device_used.append(total - free)
        if reference is not None:
            access.refresh()
            reference.check(
                model, None, torch.tensor(loss, device=case.device), expected
            )
            if evidence is None:
                evidence = access.evidence(model)
    clock.close()
    return {
        "evidence": evidence,
        "reference_checks": None if reference is None else reference.checks,
        "seconds": statistics.median(clock.cycles),
        "device_used_bytes_max": max(device_used),
    }
