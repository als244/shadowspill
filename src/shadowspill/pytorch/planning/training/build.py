"""The one-shot training plan: every phase composed, from capture to the admitted
callable."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Literal

import torch
import torch.nn as nn

from shadowspill.pipeline.common import PlanningTimer
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.planner.search import SearchOptions
from shadowspill.pytorch.planning.training.plan import plan_training_programs
from shadowspill.runtime.plan import PlanMemory
from shadowspill.step import StepDataOrdering
from shadowspill.store import ArtifactStore

from ...callables import PlannedTrainStep
from ...contracts import (
    ObjectiveResult,
)
from ...partition import (
    PartitionSpec,
)
from ..stores import open_planning_stores
from .admit import admit_training_plan, compile_selected_training_tasks
from .capture import capture_training_graphs
from .materialize import (
    materialize_training_state,
    rollback_training_failure,
    rollback_training_materialization,
)
from .profile import profile_training_tasks
from .programs import build_training_programs


def build_training(
    model: nn.Module,
    *,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    build_optimizer: Callable[[Any], torch.optim.Optimizer],
    optimizer_state_init: Callable[[str, torch.Tensor, torch.nn.Parameter], None]
    | None,
    hyperparams: Sequence[str],
    example_inputs: Sequence[Sequence[Any]],
    memory: PlanMemory,
    partition: PartitionSpec,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    data_ordering: StepDataOrdering,
    verbose: bool,
    artifact_store: ArtifactStore,
    profiling_metadata: Sequence[object] | None,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
    search_options: SearchOptions | None = None,
    incumbent: AnnotatedProgramPlan | None = None,
    transfer_bandwidths: TransferBandwidths | None = None,
) -> PlannedTrainStep:
    """Compose the independently callable training-planning boundaries.

    `incumbent` is the plan to beat for the recurrent program, and
    `transfer_bandwidths` the lanes to price copies at instead of the
    runtime's calibration, both as :func:`shadowspill.planner.plan_program`
    takes them.
    """

    started = time.perf_counter_ns()
    # Validated before any capture; the library's default when none are
    # named, which is what the store keys and the report records.
    chosen = search_options
    timer = PlanningTimer(verbose=verbose)
    artifacts = open_planning_stores(artifact_store)
    captured = capture_training_graphs(
        model,
        objective=objective,
        build_optimizer=build_optimizer,
        example_inputs=example_inputs,
        memory=memory,
        partition=partition,
        profiling_metadata=profiling_metadata,
        stores=artifacts,
        timer=timer,
    )
    materialized = materialize_training_state(
        model,
        captured,
        build_optimizer=build_optimizer,
        optimizer_state_init=optimizer_state_init,
        hyperparams=hyperparams,
        memory=memory,
        stores=artifacts,
        timer=timer,
    )
    try:
        profiled = profile_training_tasks(
            captured,
            materialized,
            plan_id=memory.plan_id,
            allocation_probe_seeds=allocation_probe_seeds,
            allocation_probe_repetitions=allocation_probe_repetitions,
            stores=artifacts,
            timer=timer,
        )
        captured = replace(captured, partitioned=profiled.partitioned)
        programs = build_training_programs(
            captured,
            materialized,
            profiled,
            memory=memory,
            optimizer_ordering=optimizer_ordering,
            data_ordering=data_ordering,
            timer=timer,
            transfer_bandwidths=transfer_bandwidths,
        )
        selections = plan_training_programs(
            programs,
            stores=artifacts,
            timer=timer,
            search_options=chosen,
            incumbent=None if incumbent is None else incumbent.result,
        )
        executable = compile_selected_training_tasks(
            profiled,
            programs,
            selections,
            installed=captured.installed,
            timer=timer,
        )
    except BaseException as error:
        rollback_training_failure(
            memory.runtime,
            error,
            lambda: rollback_training_materialization(model, materialized),
            operation="profile and lower training plan",
        )
    return admit_training_plan(
        model,
        captured,
        materialized,
        profiled,
        programs,
        selections,
        executable,
        memory=memory,
        optimizer_ordering=optimizer_ordering,
        data_ordering=data_ordering,
        stores=artifacts,
        timer=timer,
        started=started,
        search_options=chosen,
    )
