"""Step programs: one self-contained `StepProgram` per ordering, before any search --
looked up in the step archive first, and built once for every ordering that misses."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import torch
import torch.nn as nn

from shadowspill.pipeline.admission import dynamic_scratch_reserve_bytes
from shadowspill.pipeline.common import (
    PlanningTimer,
    program_phase_timings,
)
from shadowspill.pipeline.reporting import (
    cache_artifacts,
)
from shadowspill.planner import (
    AdmissionFacts,
)
from shadowspill.planner.program import ShadowSpillPlanningProblem
from shadowspill.pytorch.profiling import profile_environment
from shadowspill.pytorch.profiling.environment import DEVICE_POOL_PROVIDER_ID
from shadowspill.runtime.plan import PlanMemory
from shadowspill.simulator import SimulationConfig
from shadowspill.step import StepDataOrdering, StepProgram
from shadowspill.store import ArtifactStore

from ...contracts import (
    ObjectiveResult,
)
from ...diagnostics import PlanProfilingMetadata
from ...lowering.training import (
    LoweredTrainingProgram,
)
from ...partition import (
    PartitionSpec,
)
from ..artifacts import (
    TrainingCaptureArtifacts,
    TrainingProfileArtifacts,
    TrainingProgramArtifacts,
)
from ..identity import machine_identity, step_identity, step_key
from ..stores import PlanningStores, open_planning_stores
from .capture import capture_training_graphs
from .materialize import (
    materialize_training_state,
    rollback_training_failure,
    rollback_training_materialization,
)
from .profile import profile_training_tasks, release_build_executables
from .programs import build_training_programs


def make_training_programs(
    model: nn.Module,
    *,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    build_optimizer: Callable[[Any], torch.optim.Optimizer],
    hyperparams: Sequence[str],
    example_inputs: Sequence[Sequence[Any]],
    memory: PlanMemory,
    partition: PartitionSpec,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    data_orderings: Sequence[StepDataOrdering],
    verbose: bool,
    artifact_store: ArtifactStore,
    profiling_metadata: Sequence[object] | None,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
) -> tuple[StepProgram, ...]:
    """Build one self-contained step artifact per ordering, before any search.

    One capture, one materialisation and one profiling serve every ordering,
    which differ only in the walk the lowering emits. With an export bypass
    key, each ordering's program is looked up in the step archive first and
    only the missing ones are built; a hit's phase timings are its lookup.
    """

    started = time.perf_counter_ns()
    timer = PlanningTimer(verbose=verbose)
    artifacts = open_planning_stores(artifact_store)
    orderings = tuple(data_orderings)
    keys: dict[StepDataOrdering, str] = {}
    identity: dict[str, object] | None = None
    found: dict[StepDataOrdering, StepProgram] = {}
    bypass_key = artifacts.store.export_bypass_key
    if bypass_key is not None:
        with timer.measure("step_lookup"):
            identity = step_identity(
                model,
                objective=objective,
                build_optimizer=build_optimizer,
                hyperparams=hyperparams,
                example_inputs=example_inputs,
                partition=partition,
                profiling_metadata=profiling_metadata,
                optimizer_ordering=optimizer_ordering,
                allocation_probe_seeds=allocation_probe_seeds,
                allocation_probe_repetitions=allocation_probe_repetitions,
                export_bypass_key=bypass_key,
                machine=machine_identity(memory),
                environment=profile_environment(
                    device_ordinal=memory.execution_device,
                    provider_id=DEVICE_POOL_PROVIDER_ID,
                    export_bypass_key=bypass_key,
                ).identity(),
            )
            for ordering in orderings:
                keys[ordering] = step_key(identity, ordering)
                archived = artifacts.steps.read(keys[ordering])
                if archived is not None:
                    found[ordering] = archived
        # One lookup served every program found, so its wall clock is
        # charged to the first of them and the rest carry nothing, the way a
        # build charges its shared phases to the first program it makes.
        lookup_ns = timer.values[-1][1]
        for index, ordering in enumerate(item for item in orderings if item in found):
            charged = lookup_ns if index == 0 else 0
            found[ordering] = replace(
                found[ordering],
                phase_timings_ns=(("step_lookup", charged), ("total", charged)),
            )
    missing = tuple(item for item in orderings if item not in found)
    if missing:
        for ordering in missing:
            artifacts.store.build_policy.refuse_miss("step program", ordering.label)
        found.update(
            _build_training_step_programs(
                model,
                missing,
                objective=objective,
                build_optimizer=build_optimizer,
                hyperparams=hyperparams,
                example_inputs=example_inputs,
                memory=memory,
                partition=partition,
                optimizer_ordering=optimizer_ordering,
                artifacts=artifacts,
                profiling_metadata=profiling_metadata,
                allocation_probe_seeds=allocation_probe_seeds,
                allocation_probe_repetitions=allocation_probe_repetitions,
                timer=timer,
                started=started,
                keys=keys,
                identity=identity,
            )
        )
    return tuple(found[item] for item in orderings)


def _build_training_step_programs(
    model: nn.Module,
    orderings: Sequence[StepDataOrdering],
    *,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    build_optimizer: Callable[[Any], torch.optim.Optimizer],
    hyperparams: Sequence[str],
    example_inputs: Sequence[Sequence[Any]],
    memory: PlanMemory,
    partition: PartitionSpec,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    artifacts: PlanningStores,
    profiling_metadata: Sequence[object] | None,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
    timer: PlanningTimer,
    started: int,
    keys: Mapping[StepDataOrdering, str],
    identity: Mapping[str, object] | None,
) -> dict[StepDataOrdering, StepProgram]:
    """Capture, profile and lower once, and publish one program per ordering.

    The shared phases are charged to the first program's timings and each
    later program carries only its own lowering, so the timings of a step's
    programs add up to the build's wall clock rather than counting the
    shared work once per ordering.
    """

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
        hyperparams=hyperparams,
        memory=memory,
        stores=artifacts,
        timer=timer,
    )
    results: dict[StepDataOrdering, StepProgram] = {}
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
        with timer.measure("compilation"):
            release_build_executables(profiled, captured.installed)
        timer.attribute_compilation_and_profiling(profiled.profiler.wall_times)
        shared = tuple(timer.values)
        for index, ordering in enumerate(orderings):
            own_started = time.perf_counter_ns()
            mark = len(timer.values)
            programs = build_training_programs(
                captured,
                materialized,
                profiled,
                memory=memory,
                optimizer_ordering=optimizer_ordering,
                data_ordering=ordering,
                timer=timer,
            )
            own = tuple(timer.values[mark:])
            phases = (*shared, *own) if index == 0 else own
            elapsed = time.perf_counter_ns() - (started if index == 0 else own_started)
            program = _public_step_program(
                captured,
                profiled,
                programs,
                memory=memory,
                optimizer_ordering=optimizer_ordering,
                data_ordering=ordering,
                stores=artifacts,
                timer=timer,
                phase_timings_ns=program_phase_timings(phases, elapsed),
            )
            if identity is not None:
                with timer.measure("step_archival"):
                    artifacts.steps.write(keys[ordering], program, identity)
            results[ordering] = program
    except BaseException as error:
        rollback_training_failure(
            memory.runtime,
            error,
            lambda: rollback_training_materialization(model, materialized),
            operation="build training ShadowSpillProgram",
        )
    try:
        rollback_training_materialization(model, materialized)
    except BaseException as error:
        rollback_training_failure(
            memory.runtime,
            error,
            lambda: None,
            operation="release training ShadowSpillProgram build state",
        )
    return results


def _public_step_program(
    captured: TrainingCaptureArtifacts,
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    *,
    memory: PlanMemory,
    optimizer_ordering: str,
    data_ordering: StepDataOrdering,
    stores: PlanningStores,
    timer: PlanningTimer,
    phase_timings_ns: tuple[tuple[str, int], ...],
) -> StepProgram:
    """Archive programs and publish only stable, serializable planning facts."""

    with timer.measure("program_archival"):
        stores.store.archive_program(programs.lowered.program)
    scratch_reserve = dynamic_scratch_reserve_bytes(
        programs.measurements_by_profile,
        minimum_bytes=programs.dynamic_scratch_reserve_bytes,
    )
    return StepProgram(
        problem=_planning_problem_artifact(
            programs.lowered,
            programs.admission,
            programs.simulation_config,
            source_execution_budget_bytes=memory.execution_budget,
            maximum_execution_budget_bytes=(
                memory.execution.physical_capacity or memory.execution.capacity
            ),
            maximum_spill_budget_bytes=memory.spill.capacity,
            dynamic_scratch_reserve_bytes_=scratch_reserve,
        ),
        optimizer_ordering=optimizer_ordering,
        data_ordering=data_ordering,
        signature_digests=tuple(item.digest for item in captured.signatures),
        profiling_metadata=tuple(
            PlanProfilingMetadata(index, item.digest, item.canonical_json)
            for index, item in enumerate(captured.workloads)
        ),
        phase_timings_ns=phase_timings_ns,
        store_directories=stores.store.diagnostics(),
        cache_artifacts=cache_artifacts(stores.store),
        transfer_capabilities_json=json.dumps(
            memory.transfers.as_dict(), sort_keys=True, separators=(",", ":")
        ),
        unique_profile_count=profiled.profiles.unique_keys,
        captured_stage_count=sum(len(item.stages) for item in captured.partitioned),
    )


def _planning_problem_artifact(
    lowered: LoweredTrainingProgram,
    admission: AdmissionFacts,
    simulation_config: SimulationConfig,
    *,
    source_execution_budget_bytes: int,
    maximum_execution_budget_bytes: int,
    maximum_spill_budget_bytes: int,
    dynamic_scratch_reserve_bytes_: int,
) -> ShadowSpillPlanningProblem:
    device = simulation_config.devices[0]
    fixed_bytes = source_execution_budget_bytes - admission.pool_capacity_bytes
    object_reserve = admission.pool_capacity_bytes - device.capacity_bytes
    return ShadowSpillPlanningProblem(
        role="step",
        program=lowered.program,
        initial_residency=lowered.initial_residency,
        final_residency=lowered.final_residency,
        simulation_config=simulation_config,
        admission_facts=admission,
        source_execution_budget_bytes=source_execution_budget_bytes,
        maximum_execution_budget_bytes=maximum_execution_budget_bytes,
        maximum_spill_budget_bytes=maximum_spill_budget_bytes,
        fixed_execution_bytes=fixed_bytes,
        object_reserve_bytes=object_reserve,
        dynamic_scratch_reserve_bytes=dynamic_scratch_reserve_bytes_,
    )
