"""The common baseline clock surrounds only the selected Plan's training update."""

import statistics

import torch

from shadowspill.execution import TrainingResult
from shadowspill.training.observations import StepObservations
from shadowspill_workloads.full_model import HEAD_LOSS_METRIC

from .microbatches import split_microbatches
from .reference import create


def measure(case, clock, executor, geometry, data, stack):
    model, access, reference = create(case, executor, geometry, stack)
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
        pending = executor.run(args=args)
        result = pending.result()
        try:
            if not isinstance(result, TrainingResult):
                raise TypeError("training Plan returned an invalid result")
            observed = StepObservations.collect(
                result.losses, result.metrics, result.parameter_metrics
            )
        finally:
            executor.synchronize()
            del result, pending
        loss = float(sum(observed.losses)) * case.world
        head = sum(
            float(metrics.get(HEAD_LOSS_METRIC, part * case.world))
            if isinstance(metrics, dict)
            else float(part) * case.world
            for part, metrics in zip(observed.losses, observed.metrics, strict=True)
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
