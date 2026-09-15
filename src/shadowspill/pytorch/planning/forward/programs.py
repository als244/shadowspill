"""The forward program lowered, with the admission facts it implies."""

from shadowspill.errors import (
    CompilationError,
    PlanningError,
)
from shadowspill.ir import (
    MemoryLocation,
    SharedResidencyPolicy,
)
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.pytorch.compilation.compiler import CompiledTaskSet
from shadowspill.pytorch.profiling import (
    ResolvedTaskManifests,
)
from shadowspill.runtime import ObjectConsistency
from shadowspill.runtime.plan import PlanMemory

from ...lowering.forward import lower_partitioned_forward_program
from ..admission import (
    FixedLayoutSelection,
    SelectedAdmission,
    build_admission_facts,
    build_fixed_selected_admission,
    output_bindings_for_entrypoints,
)
from ..artifacts import (
    ForwardCaptureArtifacts,
    ForwardProfileArtifacts,
    ForwardProgramArtifacts,
)
from ..common import (
    PlanningTimer,
    build_simulation_config,
    fixed_execution_bytes,
    workspace_reserve,
)


def build_forward_program(
    captured: ForwardCaptureArtifacts,
    profiled: ForwardProfileArtifacts,
    *,
    memory: PlanMemory,
    timer: PlanningTimer,
    transfer_bandwidths: TransferBandwidths | None = None,
) -> ForwardProgramArtifacts:
    """Lower physical evidence into one canonical forward ShadowSpillProgram.

    `transfer_bandwidths` prices the simulator input at given lanes instead
    of the runtime's calibration; see `build_simulation_config`.
    """

    with timer.measure("program_lowering"):
        measurements = {
            artifact.compatibility_digest: measurement
            for artifact, measurement in zip(
                captured.tasks,
                profiled.profiles.measurements,
                strict=True,
            )
        }
        measurements_by_profile = dict(
            zip(
                profiled.profiles.key_digests,
                profiled.profiles.measurements,
                strict=True,
            )
        )
        lowered = lower_partitioned_forward_program(
            captured.fake_model,
            captured.partitioned,
            captured.tasks,
            profiled.profiles.measurements,
            storage_contracts={
                digest: manifest.storage_contract
                for digest, manifest in profiled.manifests.manifests.items()
            },
            compiled_root_allocations={
                digest: manifest.root_allocations
                for digest, manifest in profiled.manifests.manifests.items()
            },
            device_ordinal=captured.device_ordinal,
            profile_compatibility_digests=profiled.profiles.key_digests,
            public_output_locations=_shared_output_locations(captured, memory),
            shared_residency_by_root=_shared_input_residency(captured, memory),
        )
        reserve = workspace_reserve(profiled.profiles.measurements)
        simulation_config = build_simulation_config(
            memory, reserve, profiled.profiles, transfer_bandwidths=transfer_bandwidths
        )
        execution_pool_bytes = memory.execution_budget - fixed_execution_bytes(
            memory, profiled.profiles
        )
        output_bindings = output_bindings_for_entrypoints(
            lowered.program.tasks,
            lowered.entrypoints,
            {item.object_id: item.alias_group_id for item in lowered.program.objects},
        )
        admission = build_admission_facts(
            lowered.program,
            execution_pool_bytes=execution_pool_bytes,
            object_capacity_bytes=simulation_config.devices[0].capacity_bytes,
            allocation_traces_by_compatibility={
                digest: measurement.allocation_trace
                for digest, measurement in measurements_by_profile.items()
            },
            output_bindings=output_bindings,
        )
    return ForwardProgramArtifacts(
        lowered=lowered,
        measurements=measurements,
        measurements_by_profile=measurements_by_profile,
        workspace_reserve=reserve,
        dynamic_scratch_reserve_bytes=memory.dynamic_scratch_reserve_bytes,
        simulation_config=simulation_config,
        admission=admission,
    )


def _build_forward_admission(
    program: ForwardProgramArtifacts,
    selection: FixedLayoutSelection,
    timer: PlanningTimer,
) -> SelectedAdmission:
    with timer.measure("slab_admission"):
        selected = selection.result
        output_bindings = output_bindings_for_entrypoints(
            selected.program.selected_tasks(selected.selections),
            program.lowered.entrypoints,
            {item.object_id: item.alias_group_id for item in selected.program.objects},
        )
        return build_fixed_selected_admission(
            selected,
            program.measurements_by_profile,
            fixed_admission=selection.admission,
            output_bindings=output_bindings,
        )


def _verify_manifest_identity(
    resolved: ResolvedTaskManifests,
    compiled: CompiledTaskSet,
) -> None:
    for digest, manifest in compiled.manifests.items():
        expected = resolved.manifests.get(digest)
        if expected is None or (
            expected.compatibility_digest != manifest.compatibility_digest
        ):
            raise CompilationError(
                f"compiled entrypoint changed its storage contract: artifact={digest}"
            )


def _shared_output_locations(
    captured: ForwardCaptureArtifacts,
    memory: PlanMemory,
) -> dict[int, MemoryLocation]:
    """Translate selected runtime pool names into logical plan roles."""

    result: dict[int, MemoryLocation] = {}
    for output in captured.shared_outputs:
        if len(output.retain_in) != 1:
            raise PlanningError(
                "retaining one output in several pools requires an explicit "
                "mirror action, which is not yet supported"
            )
        pool = output.retain_in[0]
        if pool == memory.execution.name:
            location = MemoryLocation.DEVICE
        elif pool == memory.spill.name:
            location = MemoryLocation.SPILL
        else:
            raise PlanningError(
                f"shared output pool {pool!r} is not the selected execution "
                f"or spill pool"
            )
        result[output.public_leaf_index] = location
    return result


def _shared_input_residency(
    captured: ForwardCaptureArtifacts,
    memory: PlanMemory,
) -> dict[int, tuple[SharedResidencyPolicy, bool]]:
    """Project guaranteed execution residency into the canonical ShadowSpillProgram."""

    result: dict[int, tuple[SharedResidencyPolicy, bool]] = {}
    for item in captured.shared_inputs:
        if item.root_input_index is None:
            raise AssertionError("shared input has no resolved root position")
        if item.require_in != memory.execution.name:
            continue
        policy = (
            SharedResidencyPolicy.SHARED_WRITABLE_CAUSAL
            if item.consistency is ObjectConsistency.CAUSAL
            else SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED
        )
        result[item.root_input_index] = (
            policy,
            memory.spill.name in item.reference.retained_pools,
        )
    return result
