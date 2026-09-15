"""One forward plan, phase by phase.

The phases are `capture`, `profile`, `programs`, `plan`, `admit` and
`report`, as the training plan beside it is; `build_forward` composes them.
"""

import time
from collections.abc import Sequence
from typing import Any

import torch.nn as nn

from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.planner.search import SearchOptions
from shadowspill.store import ArtifactStore

from ...callables import PlannedForward
from ...partition import (
    PartitionSpec,
)
from ...runtime_adapter import PlanMemory
from ...sharing import (
    SharedOutput,
)
from ..artifacts import (
    ForwardCaptureArtifacts,
    ForwardProfileArtifacts,
    ForwardProgramArtifacts,
)
from ..common import (
    PlanningTimer,
)
from ..stores import open_planning_stores
from .admit import admit_forward_plan
from .capture import capture_forward_graph
from .plan import plan_forward_program
from .profile import profile_forward_tasks
from .programs import build_forward_program


def build_forward(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    memory: PlanMemory,
    partition: PartitionSpec,
    verbose: bool,
    artifact_store: ArtifactStore,
    profiling_metadata: object,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
    shared_outputs: Sequence[SharedOutput] = (),
    search_options: SearchOptions | None = None,
    transfer_bandwidths: TransferBandwidths | None = None,
) -> PlannedForward:
    """Compose the independently callable forward-planning boundaries.

    `transfer_bandwidths` is the lanes to price copies at instead of the
    runtime's calibration, as :func:`shadowspill.planner.plan_program` takes
    them.
    """

    started = time.perf_counter_ns()
    timer = PlanningTimer(verbose=verbose)
    artifacts = open_planning_stores(artifact_store)
    captured = capture_forward_graph(
        model,
        example_inputs=example_inputs,
        memory=memory,
        partition=partition,
        profiling_metadata=profiling_metadata,
        shared_outputs=shared_outputs,
        stores=artifacts,
        timer=timer,
    )
    profiled = profile_forward_tasks(
        captured,
        plan_id=memory.plan_id,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
        stores=artifacts,
        timer=timer,
    )
    program = build_forward_program(
        captured,
        profiled,
        memory=memory,
        timer=timer,
        transfer_bandwidths=transfer_bandwidths,
    )
    selected = plan_forward_program(
        program,
        search_options=search_options,
        stores=artifacts,
        timer=timer,
    )
    return admit_forward_plan(
        model,
        captured,
        profiled,
        program,
        selected,
        memory=memory,
        stores=artifacts,
        timer=timer,
        started=started,
    )


__all__ = [
    "ForwardCaptureArtifacts",
    "ForwardProfileArtifacts",
    "ForwardProgramArtifacts",
    "admit_forward_plan",
    "build_forward",
    "build_forward_program",
    "capture_forward_graph",
    "plan_forward_program",
    "profile_forward_tasks",
]
