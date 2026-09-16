"""Programs: the canonical initial and recurrent `ShadowSpillProgram` lowered from the
captured stages under one ordering, with the admission facts a search plans against."""

from __future__ import annotations

from typing import Literal

from shadowspill.errors import (
    PlanningError,
)
from shadowspill.pipeline.common import (
    PlanningTimer,
    build_simulation_config,
    fixed_execution_bytes,
    workspace_reserve,
)
from shadowspill.planner import (
    AdmissionFacts,
)
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.pytorch.lowering.program import execution_device_id
from shadowspill.pytorch.optimizer import (
    OptimizerCapture,
)
from shadowspill.pytorch.planning.admission import (
    build_admission_facts,
    output_bindings_for_entrypoints,
)
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.runtime.plan import PlanMemory
from shadowspill.step import StepDataOrdering

from ...lowering.profiles import CompiledLayoutIndex, ProfileMeasurementKey
from ...lowering.training import (
    LoweredTrainingProgram,
    TrainingStorageLayout,
    lower_partitioned_training_program,
)
from ..artifacts import (
    TrainingCaptureArtifacts,
    TrainingMaterializationArtifacts,
    TrainingProfileArtifacts,
    TrainingProgramArtifacts,
)


def build_training_programs(
    captured: TrainingCaptureArtifacts,
    materialized: TrainingMaterializationArtifacts,
    profiled: TrainingProfileArtifacts,
    *,
    memory: PlanMemory,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    data_ordering: StepDataOrdering,
    timer: PlanningTimer,
    transfer_bandwidths: TransferBandwidths | None = None,
) -> TrainingProgramArtifacts:
    """Construct canonical initial/recurrent programs from semantic and physical IR.

    `transfer_bandwidths` prices the simulator input at given lanes instead
    of the runtime's calibration; see `build_simulation_config`.
    """

    with timer.measure("program_lowering"):
        measurements, measurements_by_profile, compatibility_digests = (
            _training_measurement_maps(profiled)
        )
        initial, recurrent = _lower_optimizer_phases(
            captured,
            materialized.optimizer_capture,
            profiled,
            measurements,
            compatibility_digests,
            optimizer_ordering=optimizer_ordering,
            data_ordering=data_ordering,
        )
        _verify_provisional_layout(captured.layout, recurrent)
        _verify_optimizer_phase_identity(initial, recurrent)
        _report_training_program_inventory(recurrent, timer)
    with timer.measure("admission_facts"):
        reserve = workspace_reserve(profiled.profiles.measurements)
        simulation_config = build_simulation_config(
            memory,
            reserve,
            profiled.profiles,
            execution_device_id=execution_device_id(memory.execution_device),
            transfer_bandwidths=transfer_bandwidths,
        )
        execution_pool_bytes = memory.execution_budget - fixed_execution_bytes(
            memory, profiled.profiles
        )

        def admission_for(lowered: LoweredTrainingProgram) -> AdmissionFacts:
            return build_admission_facts(
                lowered.program,
                execution_pool_bytes=execution_pool_bytes,
                object_capacity_bytes=simulation_config.devices[0].capacity_bytes,
                allocation_traces_by_compatibility={
                    digest: measurement.allocation_trace
                    for digest, measurement in measurements_by_profile.items()
                },
                output_bindings=output_bindings_for_entrypoints(
                    lowered.program.tasks,
                    lowered.entrypoints,
                    {
                        item.object_id: item.alias_group_id
                        for item in lowered.program.objects
                    },
                ),
            )

        initial_admission = admission_for(initial)
        recurrent_admission = admission_for(recurrent)
    return TrainingProgramArtifacts(
        initial=initial,
        recurrent=recurrent,
        measurements=measurements,
        measurements_by_profile=measurements_by_profile,
        workspace_reserve=reserve,
        dynamic_scratch_reserve_bytes=memory.dynamic_scratch_reserve_bytes,
        simulation_config=simulation_config,
        initial_admission=initial_admission,
        recurrent_admission=recurrent_admission,
    )


def _training_measurement_maps(
    profiled: TrainingProfileArtifacts,
) -> tuple[
    dict[ProfileMeasurementKey, TaskMeasurement],
    dict[str, TaskMeasurement],
    dict[tuple[str, str | None], str],
]:
    measurements: dict[ProfileMeasurementKey, TaskMeasurement] = dict(
        zip(
            profiled.profile_keys,
            profiled.profiles.measurements,
            strict=True,
        )
    )
    by_profile = dict(
        zip(
            profiled.profiles.key_digests,
            profiled.profiles.measurements,
            strict=True,
        )
    )
    compatibility_digests = dict(
        zip(
            profiled.profile_keys,
            profiled.profiles.key_digests,
            strict=True,
        )
    )
    return measurements, by_profile, compatibility_digests


def _lower_optimizer_phases(
    captured: TrainingCaptureArtifacts,
    optimizer_capture: OptimizerCapture,
    profiled: TrainingProfileArtifacts,
    measurements: dict[ProfileMeasurementKey, TaskMeasurement],
    compatibility_digests: dict[tuple[str, str | None], str],
    *,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    data_ordering: StepDataOrdering,
) -> tuple[LoweredTrainingProgram, LoweredTrainingProgram]:
    layout_cache = CompiledLayoutIndex()
    storage_contracts = {
        digest: manifest.storage_contract
        for digest, manifest in profiled.manifests.manifests.items()
    }
    root_allocations = {
        digest: manifest.root_allocations
        for digest, manifest in profiled.manifests.manifests.items()
    }
    metadata_digests = tuple(item.digest for item in captured.workloads)

    def lower(phase: Literal["initial", "recurrent"]) -> LoweredTrainingProgram:
        return lower_partitioned_training_program(
            captured.fake_model,
            captured.partitioned,
            measurements,
            optimizer_capture,
            storage_contracts=storage_contracts,
            compiled_root_allocations=root_allocations,
            optimizer_phase=phase,
            optimizer_ordering=optimizer_ordering,
            data_ordering=data_ordering,
            layout_cache=layout_cache,
            profiling_metadata_digests=metadata_digests,
            profile_compatibility_digests=compatibility_digests,
        )

    return lower("initial"), lower("recurrent")


def _report_training_program_inventory(
    recurrent: LoweredTrainingProgram,
    timer: PlanningTimer,
) -> None:
    largest = max(
        recurrent.program.profiles,
        key=lambda item: item.workspace_bytes,
    )
    timer.progress(
        "recurrent ShadowSpillProgram inventory: "
        f"tasks={len(recurrent.program.tasks)}, "
        f"objects={len(recurrent.program.objects)}, "
        f"aliases={len(recurrent.program.alias_groups)}, "
        f"task_alternative_groups={len(recurrent.program.task_alternative_groups)}, "
        f"largest_workspace={largest.workspace_bytes} ({largest.profile_id})"
    )


def _verify_provisional_layout(
    layout: TrainingStorageLayout,
    lowered: LoweredTrainingProgram,
) -> None:
    expected = {item.object_id: item.alias_group_id for item in layout.program.objects}
    actual = {item.object_id: item.alias_group_id for item in lowered.program.objects}
    if any(
        actual.get(object_id) != alias_id for object_id, alias_id in expected.items()
    ):
        raise PlanningError(
            "training storage identities changed after optimizer capture"
        )


def _verify_optimizer_phase_identity(
    initial: LoweredTrainingProgram,
    recurrent: LoweredTrainingProgram,
) -> None:
    if initial.program.alias_groups != recurrent.program.alias_groups or (
        initial.program.objects != recurrent.program.objects
    ):
        raise PlanningError("optimizer phases changed storage identities")
