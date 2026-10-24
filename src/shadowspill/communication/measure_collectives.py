"""Measure requested collectives with identical repetition decisions on peers."""

from collections.abc import Mapping
from statistics import median
from typing import Any

import torch

from shadowspill.profiling.timing import collect_timing_samples, condition_task
from shadowspill.task.profiling import ProfilingOptions

from .binding.groups import GroupBindings
from .probe import CollectiveProbe
from .requests import CollectiveRequest


def measure_collectives(
    requests: tuple[CollectiveRequest, ...],
    *,
    device: torch.device,
    setup: Mapping[str, Any],
    options: ProfilingOptions,
    bindings: GroupBindings | None,
) -> tuple[dict[str, Any], ...]:
    """Groups are probed separately; disjoint-group contention is not inferred."""
    rows = []
    topology = {"node0": (0,)} if bindings is None else bindings.topology
    for request in requests:
        if len(request.ranks) == 1:
            observation: dict[str, Any] = dict(duration_seconds=0.0, source="identity")
        else:
            if bindings is None:
                raise ValueError(
                    "collective measurement needs bound distributed workers"
                )
            observation = _measure(request, device, options, bindings)
        rows.append(
            {
                **request.to_dict(),
                **observation,
                "setup": dict(setup),
                "topology": topology,
            }
        )
    return tuple(rows)


def _measure(
    request: CollectiveRequest,
    device: torch.device,
    options: ProfilingOptions,
    bindings: GroupBindings,
) -> dict[str, Any]:
    active = bindings.logical_rank in request.ranks
    probe, failure = None, None
    try:
        if active:
            probe = CollectiveProbe(
                request, bindings.for_members(request.ranks), device
            )
    except Exception as error:
        failure = str(error)
    bindings.check(failure)  # An allocation failure must reach nonmembers too.

    def sample() -> int:
        return 0 if probe is None else probe.sample()

    def decide(needed: bool, allowed: bool) -> bool:
        return bindings.continue_sampling(needed and active, allowed or not active)

    condition_task(sample, options=options, continue_while=decide)
    timing = collect_timing_samples(sample, options=options, continue_while=decide)
    local = (
        None
        if not active
        else {
            "logical_rank": bindings.logical_rank,
            "duration_seconds": median(timing.samples) / 1e9,
            "samples": len(timing.samples),
            "target_met": timing.window.target_met,
            "relative_mad": timing.relative_mad,
            "half_drift": timing.half_drift,
        }
    )
    peers = sorted(
        (row for row in bindings.gather(local) if row is not None),
        key=lambda row: row["logical_rank"],
    )
    return {
        "duration_seconds": max(row["duration_seconds"] for row in peers),
        "source": "maximum of participant median completion times",
        "backend": "nccl" if device.type == "cuda" else "gloo",
        "per_rank": peers,
        "scope": "one ordered group at a time; no disjoint-group overlap",
    }
