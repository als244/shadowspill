"""Measure a compiled task's occupancy of background-transfer resources."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import TYPE_CHECKING, Any

import torch

from shadowspill.errors import ProfilingError
from shadowspill.ir import TaskResourcePhase
from shadowspill.pytorch.distributed import current
from shadowspill.pytorch.distributed._profiling import invocation_batch

if TYPE_CHECKING:
    from .boundary import AllocatorBoundary
    from .measurement import MeasuredTask


def measured_resource_phases(
    boundary: AllocatorBoundary,
    source: MeasuredTask,
    stream: torch.cuda.Stream,
    runtime_ns: int,
) -> tuple[TaskResourcePhase, ...]:
    """Observe task collectives only when background sharded copies exist.

    No transfer runs during these isolated samples. The same named resource
    will serialize task collectives with the ordered transfer recipes later
    in simulation. Observing their actual intervals avoids reserving that
    resource for the entire optimizer task, which would forbid real overlap.
    """
    bound = current()
    if bound is None or not bound.transfer_sharding:
        return ()
    shared = {
        group.group_name
        for layout in bound.transfer_sharding.values()
        for group in (layout.group, layout.replica_group)
        if group is not None
    }
    resources = {
        name: ("sharded-collectives" if name in shared else f"collective:{alias}")
        for name, alias in bound.groups.source_names.items()
    }
    samples: list[Mapping[str, Any]] = []
    # Match ordinary timing's coordinated bursts. A host control exchange
    # before every sample introduces rank-arrival skew into NCCL's intervals;
    # for short tasks this can exceed the work being measured by orders of
    # magnitude. Prime NCCL's timing events and the instrumentation inside the
    # same burst, while retaining an allocator drain after every invocation.
    with invocation_batch():
        for iteration in range(10):
            with source.open() as task, boundary.scope(stream):
                torch.ops.shadowspill._collective_profile_begin(16384)
                try:
                    output = task()
                    del output
                finally:
                    sample = torch.ops.shadowspill._collective_profile_end()
            boundary.require_idle(problem="task collective profiling")
            if iteration >= 5:
                samples.append(sample)
    return resource_phases(samples, resources=resources, runtime_ns=runtime_ns)


def resource_phases(
    samples: Sequence[Mapping[str, Any]],
    *,
    resources: Mapping[str, str],
    runtime_ns: int,
) -> tuple[TaskResourcePhase, ...]:
    """Normalize stable median segment proportions to the uninstrumented time.

    Only sequential collective intervals are representable by this contract.
    Never silently flatten overlapping communicators or discard missing work
    events. Integer cumulative rounding preserves the exact total runtime.
    """
    if not samples or runtime_ns < 0:
        raise ProfilingError("task collective profiling needs samples and a runtime")
    trials = []
    signatures = []
    for sample in samples:
        if sample.get("errors", 0):
            raise ProfilingError("task collective timing events were incomplete")
        duration = _nanoseconds(sample["duration_ms"])
        cursor = 0.0
        phases: list[float] = []
        signature: list[tuple[str | None, str | None]] = []
        for event in sample["records"]:
            start = _nanoseconds(event["start_ms"])
            end = _nanoseconds(event["end_ms"])
            if start < cursor - 1000 or end < start or end > duration + 1000:
                raise ProfilingError(
                    "overlapping or out-of-task collective intervals cannot be "
                    "represented by sequential task resource phases"
                )
            group = event["group"]
            if group not in resources:
                raise ProfilingError(f"unregistered task collective group {group!r}")
            start, end = max(start, cursor), min(end, duration)
            phases.extend((start - cursor, max(0.0, end - start)))
            signature.extend(((None, None), (resources[group], event["operation"])))
            cursor = max(start, end)
        phases.append(max(0.0, duration - cursor))
        signature.append((None, None))
        trials.append(phases)
        signatures.append(signature)
    if any(value != signatures[0] for value in signatures):
        raise ProfilingError("task collective sequence changed across timing samples")
    if len(signatures[0]) == 1:
        return ()
    durations = [median(values) for values in zip(*trials, strict=True)]
    total = sum(durations)
    if not total or not 0.75 <= runtime_ns / total <= 1.25:
        raise ProfilingError(
            "instrumented task collective timing differs from isolated task "
            "timing by more than 25%; remeasure this unstable profile "
            f"(isolated={runtime_ns} ns, instrumented={round(total)} ns)"
        )
    result = []
    cumulative = 0.0
    assigned = 0
    for duration, (resource, _) in zip(durations, signatures[0], strict=True):
        cumulative += duration
        end_ns = round(runtime_ns * cumulative / total)
        result.append(TaskResourcePhase(end_ns - assigned, resource))
        assigned = end_ns
    return tuple(result)


def _nanoseconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProfilingError(
            "collective event time must be a finite nonnegative number"
        )
    if not math.isfinite(value) or value < 0:
        raise ProfilingError(
            "collective event time must be a finite nonnegative number"
        )
    return float(value) * 1e6
