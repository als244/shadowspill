"""Measure requested collectives with identical repetition decisions on peers."""

from collections.abc import Mapping
from statistics import median
from typing import Any

import torch

from shadowspill.profiling.timing import collect_timing_samples, condition_task
from shadowspill.task.profiling import ProfilingOptions

from .binding.groups import GroupBindings
from .memory_measurement import measure_memory
from .probe import CollectiveProbe
from .requests import CollectiveRequest


def measure_collectives(
    requests: tuple[CollectiveRequest, ...],
    *,
    device: torch.device,
    setup: Mapping[str, Any],
    options: ProfilingOptions,
    bindings: GroupBindings | None,
    backend_setup: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Groups are probed separately; disjoint-group contention is not inferred."""
    rows = []
    topology = {"node0": (0,)} if bindings is None else bindings.topology
    for request in requests:
        if len(request.ranks) == 1:
            observation: dict[str, Any] = dict(
                duration_seconds=0.0,
                source="identity",
                memory={"workspace_bytes": 0, "memory_kind": "static"},
            )
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
                "backend_setup": dict(backend_setup or {}),
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
    memory = measure_memory(probe, device, bindings)
    zero, failure = None, None
    try:
        zero = None if probe is None else probe.zero_seconds()
    except Exception as error:
        failure = str(error)
    bindings.check(failure)
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
            "memory": memory,
            "zero_duration_seconds": zero,
        }
    )
    peers = sorted(
        (row for row in bindings.gather(local) if row is not None),
        key=lambda row: row["logical_rank"],
    )
    memory_rows = [
        {"logical_rank": row["logical_rank"], **row["memory"]} for row in peers
    ]
    complete = all(row["memory_kind"] == "measured" for row in memory_rows)
    return {
        "duration_seconds": max(row["duration_seconds"] for row in peers),
        "source": "maximum of participant median completion times",
        "backend": "nccl" if device.type == "cuda" else "gloo",
        "per_rank": peers,
        "zero_duration_seconds": max(
            (
                row["zero_duration_seconds"]
                for row in peers
                if row["zero_duration_seconds"] is not None
            ),
            default=None,
        ),
        "scope": "one ordered group at a time; no disjoint-group overlap",
        "memory": {
            "memory_kind": "measured" if complete else "unresolved",
            "workspace_bytes": (
                max(row["workspace_bytes"] for row in memory_rows) if complete else None
            ),
            "per_rank": memory_rows,
        },
    }
