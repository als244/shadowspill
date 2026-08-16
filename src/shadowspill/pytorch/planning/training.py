"""Composable accumulated-training planning artifact boundaries."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, NoReturn

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.ir import EntrypointSpec, ExecutionPlan, PhysicalAdmission
from shadowspill.planner import (
    AdmissionTopology,
    PressureFitInfeasibleError,
    PressureFitResult,
    validate_schedule_feasibility,
)
from shadowspill.pytorch.capture.aot import (
    TrainingObjectiveCapture,
    capture_training_objective,
)
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.compilation.compiler import CompiledTaskSet
from shadowspill.pytorch.diagnostics.builders import training_stage_inventory
from shadowspill.pytorch.materialization.training import (
    TrainingMaterializedState,
    representative_training_arguments,
)
from shadowspill.pytorch.optimizer import (
    OptimizerCapture,
    OptimizerTaskArtifact,
    capture_optimizer,
    training_parameter_stage_owners,
)
from shadowspill.pytorch.profiling import (
    ProfileEnvironment,
    ProfilingResult,
    ResolvedTaskManifests,
    TaskMeasurement,
    profile_environment,
    profile_unique_artifacts,
    resolve_task_manifests,
    validate_compiled_profile,
)
from shadowspill.pytorch.profiling.context import profile_input_context_digest
from shadowspill.pytorch.profiling.metadata import (
    ProfilingMetadata,
    training_profiling_metadata,
)
from shadowspill.pytorch.profiling.profiler import CudaTaskProfiler
from shadowspill.pytorch.runtime_adapter.allocator import (
    InstalledAllocator,
    validate_dynamic_execution_reservation,
)
from shadowspill.pytorch.runtime_adapter.bridge import RuntimeBridge
from shadowspill.pytorch.state.optimizer import (
    release_optimizer_state_from_plan,
    relocate_optimizer_state_for_plan,
)

from ..cache import PlanningCache
from ..callables import PlannedTrainStep
from ..contracts import (
    AdmissionError,
    CompilationError,
    ObjectiveResult,
    PlanningError,
)
from ..diagnostics import PlanReport
from ..execution import TrainingExecutor
from ..graph_pairs import (
    PartitionedTrainingCapture,
    partition_training_capture,
    resolve_partitioned_saved_controls,
)
from ..guards import InputSignature, capture_training_signatures
from ..lowering.profiles import CompiledLayoutIndex, ProfileMeasurementKey
from ..lowering.training import (
    LoweredTrainingProgram,
    TrainingStorageLayout,
    lower_partitioned_training_program,
    lower_training_storage_layout,
)
from ..materialization import representative_cpu_inputs
from ..partition import (
    PartitionSpec,
)
from ..runtime_adapter import INITIAL_PLACEMENT_TASK_ID, PlanMemory, Runtime
from .admission import (
    FixedLayoutInfeasibleError,
    SelectedAdmission,
    build_admission_topology,
    build_fixed_selected_admission,
    dynamic_scratch_reserve_bytes,
    output_bindings_for_entrypoints,
    physical_admission,
    project_runtime_fixed_layout,
    reconcile_spill_pool,
    resolve_fixed_layout_selection,
    seal_physical_budget,
)
from .artifacts import (
    TrainingAdmissionArtifacts,
    TrainingCaptureArtifacts,
    TrainingExecutableArtifacts,
    TrainingMaterializationArtifacts,
    TrainingProfileArtifacts,
    TrainingProgramArtifacts,
    TrainingSelections,
)
from .common import (
    PlanningTimer,
    build_simulation_config,
    estimate_spill_reservation,
    fixed_execution_bytes,
    public_infeasible_plan_error,
    validate_budgets,
    validate_cpu_model,
    workspace_reserve,
)
from .reporting import (
    build_training_report,
    cache_artifacts,
    fixed_layout_diagnostic,
    publish_plan_report,
)
from .repositories import PlanningArtifactRepositories, open_artifact_repositories


@dataclass(frozen=True, slots=True)
class _TrainingTaskInventory:
    compile_tasks: tuple[OptimizerTaskArtifact, ...]
    profile_keys: tuple[tuple[str, str | None, str | None], ...]
    profile_tasks: tuple[OptimizerTaskArtifact, ...]
    profile_metadata_digests: tuple[str | None, ...]


def capture_training_graphs(
    model: nn.Module,
    *,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    opt: Callable[[Any], torch.optim.Optimizer],
    example_inputs: Sequence[Sequence[Any]],
    memory: PlanMemory,
    partition: PartitionSpec,
    profiling_metadata: Sequence[object] | None,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> TrainingCaptureArtifacts:
    """Capture objective and stage-local graph pairs entirely offline."""

    with timer.measure("validation"):
        signatures, cpu_inputs, workloads = _prepare_training_inputs(
            model,
            objective,
            opt,
            example_inputs,
            memory,
            profiling_metadata,
        )
    with timer.measure("runtime_binding"):
        installed = memory.installed
        device_ordinal = memory.execution_device
    with timer.measure("capture_lowering"):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_model = fake_cuda_model(model, fake_mode, device_index=device_ordinal)
        captures = _capture_training_objectives(
            fake_model,
            objective,
            cpu_inputs,
            fake_mode=fake_mode,
            device_ordinal=device_ordinal,
            artifact_cache=artifact_cache,
            timer=timer,
        )
        partitioned = _partition_training_graphs(
            model,
            captures,
            cpu_inputs,
            fake_mode=fake_mode,
            partition=partition,
            artifact_cache=artifact_cache,
            timer=timer,
        )
        with timer.measure("storage_layout_lowering"):
            layout = lower_training_storage_layout(fake_model, captures)
    return TrainingCaptureArtifacts(
        signatures,
        cpu_inputs,
        workloads,
        installed,
        device_ordinal,
        fake_model,
        captures,
        partitioned,
        layout,
    )


def _prepare_training_inputs(
    model: nn.Module,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    opt: Callable[[Any], torch.optim.Optimizer],
    example_inputs: Sequence[Sequence[Any]],
    memory: PlanMemory,
    profiling_metadata: Sequence[object] | None,
) -> tuple[
    tuple[InputSignature, ...],
    tuple[tuple[object, ...], ...],
    tuple[ProfilingMetadata, ...],
]:
    validate_cpu_model(model)
    validate_budgets(memory.execution_budget, memory.spill_budget)
    if not callable(objective):
        raise TypeError("objective must be callable")
    if not callable(opt):
        raise TypeError("opt must be an optimizer factory")
    signatures = capture_training_signatures(example_inputs)
    cpu_inputs = tuple(
        tuple(representative_cpu_inputs(microbatch)) for microbatch in example_inputs
    )
    estimate_spill_reservation(model, cpu_inputs, memory.spill_budget)
    workloads = training_profiling_metadata(
        profiling_metadata,
        microbatch_count=len(example_inputs),
    )
    return signatures, cpu_inputs, workloads


def _capture_training_objectives(
    fake_model: nn.Module,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    cpu_inputs: tuple[tuple[object, ...], ...],
    *,
    fake_mode: FakeTensorMode,
    device_ordinal: int,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> tuple[TrainingObjectiveCapture, ...]:
    with fake_mode, timer.measure("objective_export"):
        captures = tuple(
            capture_training_objective(
                fake_model,
                objective,
                fake_cuda_inputs(
                    microbatch,
                    fake_mode,
                    device_index=device_ordinal,
                ),
            )
            for microbatch in cpu_inputs
        )
    with timer.measure("export_archival"):
        for position, capture in enumerate(captures):
            artifact_cache.archive_export(
                capture.exported,
                mode="training_objective",
                position=position,
            )
    return captures


def _partition_training_graphs(
    model: nn.Module,
    captures: tuple[TrainingObjectiveCapture, ...],
    cpu_inputs: tuple[tuple[object, ...], ...],
    *,
    fake_mode: FakeTensorMode,
    partition: PartitionSpec,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> tuple[PartitionedTrainingCapture, ...]:
    representative_roots = tuple(
        representative_training_arguments(capture, model, microbatch)
        for capture, microbatch in zip(captures, cpu_inputs, strict=True)
    )
    with fake_mode, timer.measure("stage_partition_aot"):
        return tuple(
            partition_training_capture(
                capture,
                partition=partition,
                graph_pair_repository=artifact_cache.graph_pairs,
                representative_root_inputs=root_inputs,
            )
            for capture, root_inputs in zip(
                captures,
                representative_roots,
                strict=True,
            )
        )


def materialize_training_state(
    model: nn.Module,
    captured: TrainingCaptureArtifacts,
    *,
    opt: Callable[[Any], torch.optim.Optimizer],
    runtime: Runtime,
    spill_pool: str,
    timer: PlanningTimer,
) -> TrainingMaterializationArtifacts:
    """Materialize registered state and invoke/capture the optimizer exactly once."""

    bridge = RuntimeBridge(captured.installed.library, captured.layout.program)
    state: TrainingMaterializedState | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        with timer.measure("model_materialization"):
            state = TrainingMaterializedState(
                model,
                captured.layout,
                captured.captures,
                captured.cpu_inputs,
                bridge,
                runtime=runtime,
                device_ordinal=captured.device_ordinal,
            )
        with timer.measure("optimizer_capture"):
            optimizer = opt(model.parameters())
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise PlanningError("optimizer factory must return Optimizer")
            state.restore_model_cpu_for_optimizer_capture()
            optimizer_capture = capture_optimizer(
                dict(model.named_parameters()),
                optimizer,
                parameter_stage_owners=training_parameter_stage_owners(
                    captured.partitioned,
                    dict(model.named_parameters()),
                ),
            )
            if optimizer_capture.initialized_state_dict is not None:
                optimizer.load_state_dict(optimizer_capture.initialized_state_dict)
        with timer.measure("optimizer_state_relocation"):
            relocate_optimizer_state_for_plan(
                optimizer,
                runtime=runtime,
                pool=spill_pool,
            )
        with timer.measure("model_placeholder_restoration"):
            state.restore_cuda_placeholders_after_optimizer_capture()
        if optimizer_capture.recurrent is None:
            raise PlanningError(
                "the optimizer state/update cannot be bounded: "
                f"{optimizer_capture.opaque_reason}"
            )
        return TrainingMaterializationArtifacts(state, optimizer, optimizer_capture)
    except BaseException as error:
        if state is not None:
            def rollback_partial_materialization() -> None:
                _restore_training_ownership(
                    model,
                    state,
                    optimizer,
                    runtime=runtime,
                )

            _rollback_training_failure(
                runtime,
                error,
                rollback_partial_materialization,
                operation="materialize training state",
            )
        raise


def profile_training_tasks(
    captured: TrainingCaptureArtifacts,
    materialized: TrainingMaterializationArtifacts,
    *,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> TrainingProfileArtifacts:
    """Compile/profile each unique graph-pair and optimizer structural ABI."""

    profiler = CudaTaskProfiler(
        captured.installed.library,
        device_ordinal=captured.device_ordinal,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
    )
    with timer.measure("saved_control_resolution"):
        partitioned = resolve_partitioned_saved_controls(
            captured.partitioned,
            profiler.resolve_graph_pair_controls,
        )
    resolved_capture = replace(captured, partitioned=partitioned)
    inventory = _training_task_inventory(
        resolved_capture,
        materialized.optimizer_capture,
    )
    _report_training_profile_inventory(
        inventory,
        materialized.optimizer_capture,
        timer,
    )
    environment = profile_environment(
        device_ordinal=captured.device_ordinal,
        provider_id="shadowspill.device_pool",
        implementation_revision=artifact_cache.store.implementation_revision,
    )
    manifests = _resolve_training_manifests(
        inventory,
        profiler,
        environment,
        artifact_cache,
        timer,
    )
    profiles = _profile_training_inventory(
        inventory,
        profiler,
        environment,
        manifests,
        artifact_cache,
        timer,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
    )
    return TrainingProfileArtifacts(
        partitioned,
        inventory.compile_tasks,
        inventory.profile_keys,
        inventory.profile_tasks,
        inventory.profile_metadata_digests,
        profiler,
        manifests,
        profiles,
    )


def _report_training_profile_inventory(
    inventory: _TrainingTaskInventory,
    optimizer: OptimizerCapture,
    timer: PlanningTimer,
) -> None:
    optimizer_count = sum(
        not isinstance(item, GraphArtifact) or item.kind == "optimizer"
        for item in inventory.compile_tasks
    )
    timer.progress(
        "structural artifact inventory: "
        f"graph={len(inventory.compile_tasks) - optimizer_count}, "
        f"optimizer={optimizer_count}, "
        f"unique={len(inventory.compile_tasks)}, "
        f"profile_variants={len(inventory.profile_tasks)}, "
        "optimizer_tasks="
        f"{len(optimizer.recurrent_tasks)}"
    )


def _resolve_training_manifests(
    inventory: _TrainingTaskInventory,
    profiler: CudaTaskProfiler,
    environment: ProfileEnvironment,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> ResolvedTaskManifests:
    with timer.measure("compiler_manifest"):
        manifests = resolve_task_manifests(
            inventory.compile_tasks,
            environment=environment,
            profile_cache=artifact_cache.profiles,
            compiler=profiler,
            progress=lambda index, total, state, digest: timer.progress(
                f"compiled manifest {index}/{total} {state}: {digest[:12]}"
            ),
        )
        timer.progress(
            "compiled manifest cache: "
            f"hits={manifests.cache_hits}, misses={manifests.cache_misses}"
        )
    return manifests


def _profile_training_inventory(
    inventory: _TrainingTaskInventory,
    profiler: CudaTaskProfiler,
    environment: ProfileEnvironment,
    manifests: ResolvedTaskManifests,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
    *,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
) -> ProfilingResult:
    with timer.measure("structural_profiling"):
        return profile_unique_artifacts(
            inventory.profile_tasks,
            environment=environment,
            measure=profiler.measure,
            cache=artifact_cache.profiles,
            validate=lambda artifact, measurement: validate_compiled_profile(
                artifact,
                measurement,
                manifests.manifests,
            ),
            progress=lambda index, total, state, digest: timer.progress(
                f"structural profile {index}/{total} {state}: {digest[:12]}"
            ),
            profiling_metadata_digests=inventory.profile_metadata_digests,
            allocation_probe_seeds=allocation_probe_seeds,
            allocation_probe_repetitions=allocation_probe_repetitions,
        )


def build_training_programs(
    captured: TrainingCaptureArtifacts,
    materialized: TrainingMaterializationArtifacts,
    profiled: TrainingProfileArtifacts,
    *,
    memory: PlanMemory,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    timer: PlanningTimer,
) -> TrainingProgramArtifacts:
    """Construct canonical initial/recurrent Programs from semantic and physical IR."""

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
        )
        _verify_provisional_layout(captured.layout, recurrent)
        _verify_optimizer_phase_identity(initial, recurrent)
        _report_training_program_inventory(recurrent, timer)
        reserve = workspace_reserve(profiled.profiles.measurements)
        simulation_config = build_simulation_config(memory, reserve, profiled.profiles)
        execution_pool_bytes = memory.execution_budget - fixed_execution_bytes(
            memory, profiled.profiles
        )

        def admission_for(lowered: LoweredTrainingProgram) -> AdmissionTopology:
            return build_admission_topology(
                lowered.program,
                execution_pool_bytes=execution_pool_bytes,
                object_capacity_bytes=simulation_config.devices[0].capacity_bytes,
                workspace_extents_by_compatibility={
                    digest: measurement.workspace_extent_bytes
                    for digest, measurement in measurements_by_profile.items()
                },
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
    dict[tuple[str, str | None, str | None], str],
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
    compatibility_digests: dict[tuple[str, str | None, str | None], str],
    *,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
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
        "recurrent Program inventory: "
        f"tasks={len(recurrent.program.tasks)}, "
        f"objects={len(recurrent.program.objects)}, "
        f"aliases={len(recurrent.program.alias_groups)}, "
        f"recomputation_groups={len(recurrent.program.recomputation_groups)}, "
        f"largest_workspace={largest.workspace_bytes} ({largest.profile_id})"
    )


def pressurefit_training_programs(
    programs: TrainingProgramArtifacts,
    *,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> TrainingSelections:
    """Resolve recurrent and, when required, lazy-state first-step selections."""

    needs_initial = any(
        item.created_on_first_step for item in programs.initial.optimizer_objects
    )
    with timer.measure("feasibility_preflight"):
        try:
            validate_schedule_feasibility(
                programs.recurrent.program,
                initial_residency=programs.recurrent.initial_residency,
                final_residency=programs.recurrent.final_residency,
                config=programs.simulation_config,
                admission=programs.recurrent_admission,
            )
            if needs_initial:
                validate_schedule_feasibility(
                    programs.initial.program,
                    initial_residency=programs.initial.initial_residency,
                    final_residency=programs.initial.final_residency,
                    config=programs.simulation_config,
                    admission=programs.initial_admission,
                )
        except PressureFitInfeasibleError as error:
            raise public_infeasible_plan_error(error) from error
    with timer.measure("pressurefit_simulation"):
        try:
            recurrent = resolve_fixed_layout_selection(
                programs.simulation_config,
                programs.recurrent_admission,
                lambda config: artifact_cache.resolve_pressurefit(
                    programs.recurrent.program,
                    initial_residency=programs.recurrent.initial_residency,
                    final_residency=programs.recurrent.final_residency,
                    config=config,
                    progress=timer.progress,
                ),
                scratch_reserve_bytes=dynamic_scratch_reserve_bytes(
                    programs.measurements_by_profile,
                    minimum_bytes=programs.dynamic_scratch_reserve_bytes,
                ),
                progress=timer.progress,
            )
            initial = (
                resolve_fixed_layout_selection(
                    programs.simulation_config,
                    programs.initial_admission,
                    lambda config: artifact_cache.resolve_pressurefit(
                        programs.initial.program,
                        initial_residency=programs.initial.initial_residency,
                        final_residency=programs.initial.final_residency,
                        config=config,
                        progress=timer.progress,
                    ),
                    scratch_reserve_bytes=dynamic_scratch_reserve_bytes(
                        programs.measurements_by_profile,
                        minimum_bytes=programs.dynamic_scratch_reserve_bytes,
                    ),
                    progress=timer.progress,
                )
                if needs_initial
                else None
            )
        except PressureFitInfeasibleError as error:
            raise public_infeasible_plan_error(error) from error
        except FixedLayoutInfeasibleError as error:
            raise AdmissionError(f"fixed slab admission failed: {error}") from error
    return TrainingSelections(recurrent, initial)


def compile_selected_training_tasks(
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    *,
    installed: InstalledAllocator,
    timer: PlanningTimer,
) -> TrainingExecutableArtifacts:
    """Retain executable callables only for selected task variants."""

    required = _selected_artifact_digests(
        programs.recurrent,
        selections.recurrent.result,
    )
    if selections.initial is not None:
        required.update(
            _selected_artifact_digests(programs.initial, selections.initial.result)
        )
    selected_tasks = tuple(
        artifact
        for artifact in profiled.compile_tasks
        if artifact.compatibility_digest in required
    )
    with timer.measure("compilation"):
        compiled = profiled.profiler.take_compiled_tasks(
            selected_tasks,
            progress=lambda index, total, state, digest: timer.progress(
                f"selected entrypoint {index}/{total} {state}: {digest[:12]}"
            ),
        )
        _verify_compiled_manifest_identity(profiled.manifests, compiled)
        profiled.profiler.discard_compiled_tasks()
        installed.library.shadowspill_pytorch_allocator_wait_idle()
        validate_dynamic_execution_reservation(
            installed,
            reserved_bytes=(
                installed.fixed_execution_bytes + profiled.profiles.fixed_slab_bytes
            ),
        )
    timer.attribute_compilation_and_profiling(profiled.profiler)
    return TrainingExecutableArtifacts(compiled)


def admit_training_plan(
    model: nn.Module,
    captured: TrainingCaptureArtifacts,
    materialized: TrainingMaterializationArtifacts,
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    executable: TrainingExecutableArtifacts,
    *,
    memory: PlanMemory,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
    started: int,
) -> PlannedTrainStep:
    """Physically admit selections and publish the training callable/report."""

    try:
        admitted = _admit_training_execution_plans(
            captured,
            profiled,
            programs,
            selections,
            materialized.optimizer_capture,
            memory,
            timer,
        )
        recurrent_plan = admitted.recurrent
        initial_plan = admitted.initial
        recurrent_fixed_layout = admitted.recurrent_admission.fixed_layout
        if recurrent_fixed_layout is None:
            raise AssertionError("recurrent admission did not produce a fixed layout")
        recurrent_runtime_layout = project_runtime_fixed_layout(
            recurrent_fixed_layout,
            recurrent_plan.program,
            recurrent_plan.schedule,
            initial_task_id=INITIAL_PLACEMENT_TASK_ID,
            dynamic_task_allocations=(
                admitted.recurrent_admission.dynamic_provider_allocations()
            ),
        )
        initial_runtime_layout = None
        if initial_plan is not None:
            initial_admission = admitted.initial_admission
            if initial_admission is None or initial_admission.fixed_layout is None:
                raise AssertionError("initial admission did not produce a fixed layout")
            initial_runtime_layout = project_runtime_fixed_layout(
                initial_admission.fixed_layout,
                initial_plan.program,
                initial_plan.schedule,
                initial_task_id=INITIAL_PLACEMENT_TASK_ID,
                dynamic_task_allocations=(
                    initial_admission.dynamic_provider_allocations()
                ),
            )
        bridge = RuntimeBridge(captured.installed.library, recurrent_plan.program)
        with timer.measure("plan_adoption"):
            materialized.state.adopt_execution_plan(
                bridge,
                programs.recurrent,
                optimizer=materialized.optimizer,
            )
        with timer.measure("physical_sealing"):
            seal_physical_budget(captured.installed, recurrent_plan)
        with timer.measure("callable_construction"):
            executor = TrainingExecutor(
                None if initial_plan is None else (programs.initial, initial_plan),
                (programs.recurrent, recurrent_plan),
                bridge,
                materialized.state,
                executable.tasks.functions,
                materialized.optimizer,
                recurrent_simulation=admitted.recurrent_admission.simulation,
                initial_simulation=(
                    None
                    if admitted.initial_admission is None
                    else admitted.initial_admission.simulation
                ),
                initial_fixed_layout=initial_runtime_layout,
                recurrent_fixed_layout=recurrent_runtime_layout,
                initial_memory_envelopes=(
                    None
                    if admitted.initial_admission is None
                    else admitted.initial_admission.envelopes_by_task()
                ),
                recurrent_memory_envelopes=(
                    admitted.recurrent_admission.envelopes_by_task()
                ),
                optimizer_state_preinitialized=(
                    materialized.optimizer_capture.initialized_state_dict is not None
                    or bool(
                        materialized.optimizer_capture.preinitialized_state_names
                    )
                ),
                optimizer_state_was_lazy=bool(
                    materialized.optimizer_capture.created_state_names
                ),
            )
        report = _training_plan_report(
            model,
            captured,
            profiled,
            programs,
            selections,
            admitted,
            recurrent_plan,
            initial_plan,
            optimizer_ordering=optimizer_ordering,
            artifact_cache=artifact_cache,
            memory=memory,
            timer=timer,
            started=started,
        )
        return PlannedTrainStep(
            model,
            captured.signatures,
            executor,
            materialized.state,
            materialized.optimizer,
            report,
            memory.runtime,
        )
    except BaseException as error:
        _rollback_training_failure(
            memory.runtime,
            error,
            lambda: rollback_training_materialization(model, materialized),
            operation="admit training plan",
        )


def _admit_training_execution_plans(
    captured: TrainingCaptureArtifacts,
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    optimizer_capture: OptimizerCapture,
    memory: PlanMemory,
    timer: PlanningTimer,
) -> TrainingAdmissionArtifacts:
    recurrent = selections.recurrent.result
    initial = None if selections.initial is None else selections.initial.result
    with timer.measure("host_admission"):
        reconcile_spill_pool(
            predicted_peak=max(
                recurrent.simulation.host_peak_bytes,
                0 if initial is None else initial.simulation.host_peak_bytes,
            ),
            budget=memory.spill_budget,
        )
    admissions = _build_training_admissions(
        captured,
        profiled,
        programs,
        selections,
        memory,
        timer,
    )
    admission = physical_admission(
        memory,
        captured.installed,
        workspace_reserve=programs.workspace_reserve,
        predicted_fragmentation_bytes=max(
            item.predicted_fragmentation_bytes for item in admissions
        ),
    )
    recurrent_plan = _execution_plan(
        programs.recurrent,
        admissions[0].apply_prediction(recurrent),
        optimizer_capture.optimizer_type,
        admission,
    )
    initial_plan = (
        _execution_plan(
            programs.initial,
            admissions[1].apply_prediction(initial),
            optimizer_capture.optimizer_type,
            admission,
        )
        if initial is not None
        else None
    )
    recurrent_admission = admissions[0]
    initial_admission = admissions[1] if len(admissions) == 2 else None
    recurrent_result = recurrent_admission.apply_prediction(recurrent)
    initial_result = (
        None
        if initial is None or initial_admission is None
        else initial_admission.apply_prediction(initial)
    )
    return TrainingAdmissionArtifacts(
        recurrent_plan,
        initial_plan,
        recurrent_admission,
        initial_admission,
        recurrent_result,
        initial_result,
    )


def _training_plan_report(
    model: nn.Module,
    captured: TrainingCaptureArtifacts,
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    admitted: TrainingAdmissionArtifacts,
    recurrent_plan: ExecutionPlan,
    initial_plan: ExecutionPlan | None,
    *,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    artifact_cache: PlanningArtifactRepositories,
    memory: PlanMemory,
    timer: PlanningTimer,
    started: int,
) -> PlanReport:
    with timer.measure("diagnostic_inventory"):
        task_stage_map, unique_stages = training_stage_inventory(
            captured.partitioned,
            programs.recurrent,
            recurrent_plan,
            programs.measurements,
            profiled.manifests.manifests,
            profiling_metadata_digests=tuple(
                item.digest for item in captured.workloads
            ),
        )
    hits = int(selections.recurrent.cache_hit) + (
        0 if selections.initial is None else int(selections.initial.cache_hit)
    )
    misses = int(not selections.recurrent.cache_hit) + (
        0 if selections.initial is None else int(not selections.initial.cache_hit)
    )
    report = build_training_report(
        tuple(signature.digest for signature in captured.signatures),
        recurrent_plan,
        profiled.profiles,
        tuple(timer.values),
        started,
        initial_execution_plan=initial_plan,
        recomputation_cache_hits=hits,
        recomputation_cache_misses=misses,
        captured_stage_count=sum(
            len(capture.stages) for capture in captured.partitioned
        ),
        aot_unique_stage_abis=artifact_cache.graph_pairs.unique_keys,
        aot_graph_pair_cache_hits=artifact_cache.graph_pairs.hits,
        aot_graph_pair_cache_misses=artifact_cache.graph_pairs.misses,
        pressurefit_results=(
            (admitted.recurrent_result,)
            if admitted.initial_result is None
            else (admitted.initial_result, admitted.recurrent_result)
        ),
        task_stage_map=task_stage_map,
        unique_stages=unique_stages,
        compiler_phase_timings_ns=profiled.profiler.compilation_phase_timings_ns,
        compiler_phase_timings_by_abi=(
            profiled.profiler.compilation_phase_timings_by_abi
        ),
        cache_directories=artifact_cache.store.diagnostics(),
        touched_cache_artifacts=cache_artifacts(artifact_cache.store),
        profiling_metadata=captured.workloads,
        physical_layouts=(
            *(
                ()
                if selections.initial is None or admitted.initial_admission is None
                else (
                    fixed_layout_diagnostic(
                        "initial",
                        selections.initial,
                        admitted.initial_admission,
                    ),
                )
            ),
            fixed_layout_diagnostic(
                "recurrent",
                selections.recurrent,
                admitted.recurrent_admission,
            ),
        ),
        optimizer_ordering=optimizer_ordering,
        memory=memory,
    )
    return publish_plan_report(
        model,
        report,
        artifact_cache.store,
        started=started,
    )


def rollback_training_materialization(
    model: nn.Module,
    materialized: TrainingMaterializationArtifacts,
) -> None:
    """Restore CPU ownership when a caller abandons an intermediate plan."""

    _restore_training_ownership(
        model,
        materialized.state,
        materialized.optimizer,
        runtime=materialized.state.runtime,
    )


def _restore_training_ownership(
    model: nn.Module,
    state: TrainingMaterializedState,
    optimizer: torch.optim.Optimizer | None,
    *,
    runtime: Runtime,
) -> None:
    """Attempt optimizer and model restoration even if either cleanup fails."""

    for parameter in model.parameters():
        parameter.grad = None
    release_error: BaseException | None = None
    if optimizer is not None:
        try:
            release_optimizer_state_from_plan(optimizer, runtime=runtime)
        except BaseException as error:
            release_error = error
        optimizer.state.clear()
    try:
        state.restore_cpu_and_unregister()
    except BaseException as error:
        if release_error is not None:
            release_error.add_note(f"Model-state rollback also failed: {error}")
        else:
            raise
    if release_error is not None:
        raise release_error


def _rollback_training_failure(
    runtime: Runtime,
    error: BaseException,
    rollback: Callable[[], None],
    *,
    operation: str,
) -> NoReturn:
    """Recover a planning OOM before releasing materialized frontend state."""

    runtime._prepare_failure_cleanup(
        error,
        operation=operation,
        synchronize_unlatched=False,
    )
    try:
        rollback()
    except BaseException as cleanup_error:
        error.add_note(
            f"Failed to roll back materialized training state: {cleanup_error}"
        )
    raise error


def build_training(
    model: nn.Module,
    *,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    opt: Callable[[Any], torch.optim.Optimizer],
    example_inputs: Sequence[Sequence[Any]],
    memory: PlanMemory,
    partition: PartitionSpec,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    verbose: bool,
    planning_cache: PlanningCache,
    profiling_metadata: Sequence[object] | None,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
) -> PlannedTrainStep:
    """Compose the independently callable training-planning boundaries."""

    started = time.perf_counter_ns()
    timer = PlanningTimer(verbose=verbose)
    artifacts = open_artifact_repositories(planning_cache)
    captured = capture_training_graphs(
        model,
        objective=objective,
        opt=opt,
        example_inputs=example_inputs,
        memory=memory,
        partition=partition,
        profiling_metadata=profiling_metadata,
        artifact_cache=artifacts,
        timer=timer,
    )
    materialized = materialize_training_state(
        model,
        captured,
        opt=opt,
        runtime=memory.runtime,
        spill_pool=memory.spill.name,
        timer=timer,
    )
    try:
        profiled = profile_training_tasks(
            captured,
            materialized,
            allocation_probe_seeds=allocation_probe_seeds,
            allocation_probe_repetitions=allocation_probe_repetitions,
            artifact_cache=artifacts,
            timer=timer,
        )
        captured = replace(captured, partitioned=profiled.partitioned)
        programs = build_training_programs(
            captured,
            materialized,
            profiled,
            memory=memory,
            optimizer_ordering=optimizer_ordering,
            timer=timer,
        )
        selections = pressurefit_training_programs(
            programs,
            artifact_cache=artifacts,
            timer=timer,
        )
        executable = compile_selected_training_tasks(
            profiled,
            programs,
            selections,
            installed=captured.installed,
            timer=timer,
        )
    except BaseException as error:
        _rollback_training_failure(
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
        artifact_cache=artifacts,
        timer=timer,
        started=started,
    )


def _training_task_inventory(
    captured: TrainingCaptureArtifacts,
    optimizer_capture: OptimizerCapture,
) -> _TrainingTaskInventory:
    compile_by_digest: dict[str, OptimizerTaskArtifact] = {}
    profile_by_key: dict[
        tuple[str, str | None, str | None], OptimizerTaskArtifact
    ] = {}
    for position, partitioned in enumerate(captured.partitioned):
        metadata_digest = captured.workloads[position].digest
        for stage in partitioned.stages:
            for option in stage.graph_pairs.variants:
                for artifact in (option.pair.forward, option.pair.backward):
                    compile_by_digest.setdefault(
                        artifact.compatibility_digest,
                        artifact,
                    )
                    profile_by_key.setdefault(
                        (
                            artifact.compatibility_digest,
                            metadata_digest,
                            profile_input_context_digest(artifact),
                        ),
                        artifact,
                    )
    for task in optimizer_capture.recurrent_tasks:
        compile_by_digest.setdefault(
            task.artifact.compatibility_digest,
            task.artifact,
        )
        profile_by_key.setdefault(
            (
                task.artifact.compatibility_digest,
                None,
                profile_input_context_digest(task.artifact),
            ),
            task.artifact,
        )
    if optimizer_capture.initial is not None:
        compile_by_digest.setdefault(
            optimizer_capture.initial.compatibility_digest,
            optimizer_capture.initial,
        )
        profile_by_key.setdefault(
            (
                optimizer_capture.initial.compatibility_digest,
                None,
                profile_input_context_digest(optimizer_capture.initial),
            ),
            optimizer_capture.initial,
        )
    keys = tuple(profile_by_key)
    return _TrainingTaskInventory(
        tuple(compile_by_digest.values()),
        keys,
        tuple(profile_by_key.values()),
        tuple(key[1] for key in keys),
    )


def _build_training_admissions(
    captured: TrainingCaptureArtifacts,
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    memory: PlanMemory,
    timer: PlanningTimer,
) -> tuple[SelectedAdmission, ...]:
    pairs = [
        (programs.recurrent, selections.recurrent)
    ]
    if selections.initial is not None:
        pairs.append((programs.initial, selections.initial))
    with timer.measure("slab_admission"):
        admitted: list[SelectedAdmission] = []
        for lowered, fixed_selection in pairs:
            selected = fixed_selection.result
            output_bindings = output_bindings_for_entrypoints(
                selected.program.selected_tasks(selected.selections),
                lowered.entrypoints,
                {
                    item.object_id: item.alias_group_id
                    for item in selected.program.objects
                },
            )
            admitted.append(
                build_fixed_selected_admission(
                    selected,
                    programs.measurements_by_profile,
                    fixed_admission=fixed_selection.admission,
                    output_bindings=output_bindings,
                ),
            )
        return tuple(admitted)


def _selected_artifact_digests(
    lowered: LoweredTrainingProgram,
    selected: PressureFitResult,
) -> set[str]:
    selected_task_ids = {
        task.task_id for task in lowered.program.selected_tasks(selected.selections)
    }
    return {
        entrypoint.artifact.compatibility_digest
        for entrypoint in lowered.entrypoints
        if entrypoint.task_id in selected_task_ids and entrypoint.artifact is not None
    }


def _verify_compiled_manifest_identity(
    planned: ResolvedTaskManifests,
    executable: CompiledTaskSet,
) -> None:
    for digest, manifest in executable.manifests.items():
        expected = planned.manifests.get(digest)
        if expected is None or (
            expected.compatibility_digest != manifest.compatibility_digest
        ):
            raise CompilationError(
                "selected compiled entrypoint changed its storage ABI: "
                f"artifact={digest}"
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


def _execution_plan(
    lowered: LoweredTrainingProgram,
    selected: PressureFitResult,
    optimizer_type: str,
    admission: PhysicalAdmission,
) -> ExecutionPlan:
    selected_ids = {
        task.task_id for task in lowered.program.selected_tasks(selected.selections)
    }
    active_entrypoints = tuple(
        item for item in lowered.entrypoints if item.task_id in selected_ids
    )
    return selected.to_execution_plan(
        entrypoints=tuple(
            EntrypointSpec(
                item.task_id,
                f"entrypoint_{index:06d}",
                "pytorch_inductor"
                if item.phase != "optimizer"
                else "pytorch_optimizer",
                item.artifact.compatibility_digest
                if item.artifact is not None
                else optimizer_type,
            )
            for index, item in enumerate(active_entrypoints)
        ),
        admission=admission,
    )


__all__ = [
    "TrainingCaptureArtifacts",
    "TrainingExecutableArtifacts",
    "TrainingMaterializationArtifacts",
    "TrainingProfileArtifacts",
    "TrainingProgramArtifacts",
    "TrainingSelections",
    "admit_training_plan",
    "build_training",
    "build_training_programs",
    "capture_training_graphs",
    "compile_selected_training_tasks",
    "materialize_training_state",
    "pressurefit_training_programs",
    "profile_training_tasks",
    "rollback_training_materialization",
]
