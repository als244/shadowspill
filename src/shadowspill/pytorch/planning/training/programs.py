"""Programs: the step's canonical `ShadowSpillProgram` lowered from the captured stages
under one ordering, with the admission facts a search plans against."""

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
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.pytorch.lowering.program import execution_device_id
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
    TrainingLowering,
    TrainingStorageLayout,
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
    lowering: TrainingLowering | None = None,
) -> TrainingProgramArtifacts:
    """Construct the step's canonical program from semantic and physical IR.

    `transfer_bandwidths` prices the simulator input at given lanes instead
    of the runtime's calibration; see `build_simulation_config`. `lowering`
    is the step's lowering shared by every ordering built from these
    artifacts (`training_lowering`); without one, this ordering's program is
    lowered alone.
    """

    with timer.measure("program_lowering"):
        measurements, measurements_by_profile, _ = _training_measurement_maps(profiled)
        if lowering is None:
            lowering = training_lowering(
                captured,
                materialized,
                profiled,
                optimizer_ordering=optimizer_ordering,
            )
        lowered = lowering.lower(data_ordering)
        _verify_provisional_layout(captured.layout, lowered)
        _report_training_program_inventory(lowered, timer)
    with timer.measure("admission_facts"):
        reserve = workspace_reserve(
            profiled.profiles.measurements, shared_slab=memory.slab_owner is not None
        )
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
        admission = build_admission_facts(
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
    return TrainingProgramArtifacts(
        lowered=lowered,
        measurements=measurements,
        measurements_by_profile=measurements_by_profile,
        workspace_reserve=reserve,
        dynamic_scratch_reserve_bytes=memory.dynamic_scratch_reserve_bytes,
        simulation_config=simulation_config,
        admission=admission,
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


def training_lowering(
    captured: TrainingCaptureArtifacts,
    materialized: TrainingMaterializationArtifacts,
    profiled: TrainingProfileArtifacts,
    *,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
) -> TrainingLowering:
    """The lowering every ordering of these artifacts shares: what does not
    depend on the ordering is done once, by the first program built."""

    measurements, _, compatibility_digests = _training_measurement_maps(profiled)
    return TrainingLowering(
        captured.fake_model,
        captured.partitioned,
        measurements,
        materialized.optimizer_capture,
        storage_contracts={
            digest: manifest.storage_contract
            for digest, manifest in profiled.manifests.manifests.items()
        },
        compiled_root_allocations={
            digest: manifest.root_allocations
            for digest, manifest in profiled.manifests.manifests.items()
        },
        device_ordinal=captured.device_ordinal,
        optimizer_ordering=optimizer_ordering,
        layout_cache=CompiledLayoutIndex(),
        profiling_metadata_digests=tuple(item.digest for item in captured.workloads),
        profile_compatibility_digests=compatibility_digests,
    )


def _report_training_program_inventory(
    lowered: LoweredTrainingProgram,
    timer: PlanningTimer,
) -> None:
    largest = max(
        lowered.program.profiles,
        key=lambda item: item.workspace_bytes,
    )
    timer.progress(
        "step ShadowSpillProgram inventory: "
        f"tasks={len(lowered.program.tasks)}, "
        f"objects={len(lowered.program.objects)}, "
        f"aliases={len(lowered.program.alias_groups)}, "
        f"task_alternative_groups={len(lowered.program.task_alternative_groups)}, "
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
