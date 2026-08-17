"""Composable forward-planning artifact boundaries."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, NoReturn

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import TreeSpec, tree_flatten

from shadowspill.ir import EntrypointSpec, ExecutionPlan, PhysicalAdmission
from shadowspill.planner import (
    PressureFitInfeasibleError,
    PressureFitResult,
    PressureFitSearchExhaustedError,
    validate_schedule_feasibility,
)
from shadowspill.pytorch.capture.aot import ExportCapture, capture_forward
from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    capture_forward_stage_artifacts,
)
from shadowspill.pytorch.capture.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.compilation.compiler import CompiledTaskSet
from shadowspill.pytorch.diagnostics.builders import forward_stage_inventory
from shadowspill.pytorch.profiling import (
    ResolvedTaskManifests,
    profile_environment,
    profile_unique_artifacts,
    resolve_task_manifests,
    validate_compiled_profile,
)
from shadowspill.pytorch.profiling.metadata import (
    ProfilingMetadata,
    canonicalize_profiling_metadata,
)
from shadowspill.pytorch.profiling.profiler import CudaTaskProfiler
from shadowspill.pytorch.runtime_adapter.allocator import (
    validate_dynamic_execution_reservation,
)
from shadowspill.pytorch.runtime_adapter.bridge import RuntimeBridge

from ..cache import PlanningCache
from ..callables import PlannedForward
from ..contracts import AdmissionError, CaptureError, CompilationError, PlanningError
from ..diagnostics import PlanReport
from ..execution import ForwardExecutor
from ..guards import InputSignature, capture_input_signature
from ..lowering.forward import LoweredForwardProgram, lower_partitioned_forward_program
from ..materialization import (
    MaterializedForwardState,
    flat_runtime_arguments,
    representative_cpu_inputs,
)
from ..partition import (
    PartitionedExport,
    PartitionSpec,
    partition_export,
)
from ..runtime_adapter import INITIAL_PLACEMENT_TASK_ID, PlanMemory, Runtime
from .admission import (
    FixedLayoutInfeasibleError,
    FixedLayoutSelection,
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
    ForwardCaptureArtifacts,
    ForwardProfileArtifacts,
    ForwardProgramArtifacts,
)
from .common import (
    PlanningTimer,
    build_simulation_config,
    estimate_spill_reservation,
    fixed_execution_bytes,
    public_infeasible_plan_error,
    public_search_exhausted_error,
    validate_budgets,
    validate_cpu_model,
    workspace_reserve,
)
from .reporting import (
    build_forward_report,
    cache_artifacts,
    fixed_layout_diagnostic,
    publish_plan_report,
)
from .repositories import PlanningArtifactRepositories, open_artifact_repositories


def capture_forward_graph(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    memory: PlanMemory,
    partition: PartitionSpec,
    profiling_metadata: object,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> ForwardCaptureArtifacts:
    """Validate and capture one forward graph without numerical CUDA execution."""

    with timer.measure("validation"):
        signature, cpu_inputs, workload = _prepare_forward_inputs(
            model,
            example_inputs,
            memory,
            profiling_metadata,
        )
    with timer.measure("runtime_binding"):
        installed = memory.installed
        device_ordinal = memory.execution_device
    with timer.measure("capture_lowering"):
        (
            fake_model,
            capture,
            partitioned,
            tasks,
            output_tree_spec,
        ) = _capture_partitioned_forward(
            model,
            cpu_inputs,
            device_ordinal=device_ordinal,
            partition=partition,
            artifact_cache=artifact_cache,
            timer=timer,
        )
    return ForwardCaptureArtifacts(
        signature,
        cpu_inputs,
        workload,
        installed,
        device_ordinal,
        fake_model,
        capture,
        partitioned,
        tasks,
        output_tree_spec,
    )


def _prepare_forward_inputs(
    model: nn.Module,
    example_inputs: Sequence[Any],
    memory: PlanMemory,
    profiling_metadata: object,
) -> tuple[InputSignature, tuple[object, ...], ProfilingMetadata]:
    validate_cpu_model(model)
    validate_budgets(memory.execution_budget, memory.spill_budget)
    if not isinstance(example_inputs, list | tuple):
        raise PlanningError("example_inputs must be a list or tuple")
    signature = capture_input_signature(example_inputs)
    cpu_inputs = tuple(representative_cpu_inputs(example_inputs))
    estimate_spill_reservation(model, cpu_inputs, memory.spill_budget)
    return (
        signature,
        cpu_inputs,
        canonicalize_profiling_metadata(profiling_metadata),
    )


def _capture_partitioned_forward(
    model: nn.Module,
    cpu_inputs: tuple[object, ...],
    *,
    device_ordinal: int,
    partition: PartitionSpec,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> tuple[
    nn.Module,
    ExportCapture,
    PartitionedExport,
    tuple[GraphArtifact, ...],
    TreeSpec,
]:
    try:
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_model = fake_cuda_model(model, fake_mode, device_index=device_ordinal)
        fake_inputs = fake_cuda_inputs(
            cpu_inputs,
            fake_mode,
            device_index=device_ordinal,
        )
        with fake_mode, torch.no_grad():
            output_leaves, output_tree_spec = tree_flatten(fake_model(*fake_inputs))
            del output_leaves
            capture = capture_forward(fake_model, fake_inputs)
        with timer.measure("export_archival"):
            artifact_cache.archive_export(capture, mode="forward", position=0)
        representative_roots = tuple(
            value.detach() if isinstance(value, torch.Tensor) else value
            for value in flat_runtime_arguments(capture, model, cpu_inputs)
        )
        with fake_mode, torch.no_grad():
            partitioned = partition_export(
                capture,
                fake_model,
                partition=partition,
                representative_root_inputs=representative_roots,
            )
            tasks = capture_forward_stage_artifacts(partitioned)
    except CaptureError:
        raise
    except BaseException as error:
        raise CaptureError(f"forward graph capture failed: {error}") from error
    return fake_model, capture, partitioned, tasks, output_tree_spec


def profile_forward_tasks(
    captured: ForwardCaptureArtifacts,
    *,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> ForwardProfileArtifacts:
    """Compile and profile every unique structural task contract exactly once."""

    profiler = CudaTaskProfiler(
        captured.installed.library,
        device_ordinal=captured.device_ordinal,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
    )
    environment = profile_environment(
        device_ordinal=captured.device_ordinal,
        provider_id="shadowspill.device_pool",
        implementation_revision=artifact_cache.store.implementation_revision,
    )
    with timer.measure("compiler_manifest"):
        manifests = resolve_task_manifests(
            captured.tasks,
            environment=environment,
            profile_cache=artifact_cache.profiles,
            compiler=profiler,
            progress=lambda index, total, state, digest: timer.progress(
                f"compiled manifest {index}/{total} {state}: {digest[:12]}"
            ),
        )
    with timer.measure("structural_profiling"):
        profiles = profile_unique_artifacts(
            captured.tasks,
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
            profiling_metadata_digests=(captured.workload.digest,)
            * len(captured.tasks),
            allocation_probe_seeds=allocation_probe_seeds,
            allocation_probe_repetitions=allocation_probe_repetitions,
        )
    with timer.measure("compilation"):
        compiled_tasks = profiler.take_compiled_tasks(
            captured.tasks,
            progress=lambda index, total, state, digest: timer.progress(
                f"compiled entrypoint {index}/{total} {state}: {digest[:12]}"
            ),
        )
        _verify_manifest_identity(manifests, compiled_tasks)
        captured.installed.library.shadowspill_pytorch_allocator_wait_idle()
        validate_dynamic_execution_reservation(
            captured.installed,
            reserved_bytes=(
                captured.installed.fixed_execution_bytes + profiles.fixed_slab_bytes
            ),
        )
    timer.attribute_compilation_and_profiling(profiler)
    return ForwardProfileArtifacts(profiler, manifests, profiles, compiled_tasks)


def build_forward_program(
    captured: ForwardCaptureArtifacts,
    profiled: ForwardProfileArtifacts,
    *,
    memory: PlanMemory,
    timer: PlanningTimer,
) -> ForwardProgramArtifacts:
    """Lower compiled physical evidence into one canonical forward Program."""

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
        )
        reserve = workspace_reserve(profiled.profiles.measurements)
        simulation_config = build_simulation_config(memory, reserve, profiled.profiles)
        execution_pool_bytes = memory.execution_budget - fixed_execution_bytes(
            memory, profiled.profiles
        )
        output_bindings = output_bindings_for_entrypoints(
            lowered.program.tasks,
            lowered.entrypoints,
            {item.object_id: item.alias_group_id for item in lowered.program.objects},
        )
        admission = build_admission_topology(
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


def pressurefit_forward_program(
    program: ForwardProgramArtifacts,
    *,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
) -> FixedLayoutSelection:
    """Resolve the exact PressureFit result for a canonical forward Program."""

    with timer.measure("feasibility_preflight"):
        try:
            validate_schedule_feasibility(
                program.lowered.program,
                initial_residency=program.lowered.initial_residency,
                final_residency=program.lowered.final_residency,
                config=program.simulation_config,
                admission=program.admission,
            )
        except PressureFitInfeasibleError as error:
            raise public_infeasible_plan_error(error) from error
        except PressureFitSearchExhaustedError as error:
            raise public_search_exhausted_error(error) from error
    with timer.measure("pressurefit_simulation"):
        try:
            return resolve_fixed_layout_selection(
                program.simulation_config,
                program.admission,
                lambda config: artifact_cache.resolve_pressurefit(
                    program.lowered.program,
                    initial_residency=program.lowered.initial_residency,
                    final_residency=program.lowered.final_residency,
                    config=config,
                ),
                scratch_reserve_bytes=dynamic_scratch_reserve_bytes(
                    program.measurements_by_profile,
                    minimum_bytes=program.dynamic_scratch_reserve_bytes,
                ),
                progress=timer.progress,
            )
        except PressureFitInfeasibleError as error:
            raise public_infeasible_plan_error(error) from error
        except PressureFitSearchExhaustedError as error:
            raise public_search_exhausted_error(error) from error
        except FixedLayoutInfeasibleError as error:
            raise AdmissionError(f"fixed slab admission failed: {error}") from error


def admit_forward_plan(
    model: nn.Module,
    captured: ForwardCaptureArtifacts,
    profiled: ForwardProfileArtifacts,
    program: ForwardProgramArtifacts,
    selection: FixedLayoutSelection,
    *,
    memory: PlanMemory,
    artifact_cache: PlanningArtifactRepositories,
    timer: PlanningTimer,
    started: int,
) -> PlannedForward:
    """Physically admit a selection and publish the executable callable/report."""

    selected = selection.result
    with timer.measure("host_admission"):
        reconcile_spill_pool(
            predicted_peak=selected.simulation.host_peak_bytes,
            budget=memory.spill_budget,
        )
    selected_admission = _build_forward_admission(
        program,
        selection,
        timer,
    )
    admission = physical_admission(
        memory,
        captured.installed,
        workspace_reserve=program.workspace_reserve,
        predicted_host_peak_bytes=selected.simulation.host_peak_bytes,
        predicted_fragmentation_bytes=(
            selected_admission.predicted_fragmentation_bytes
        ),
    )
    admitted_result = selected_admission.apply_prediction(selected)
    execution_plan = _forward_execution_plan(
        program.lowered,
        admitted_result,
        admission,
    )
    fixed_layout = selected_admission.fixed_layout
    if fixed_layout is None:
        raise AssertionError("forward admission did not produce a fixed layout")
    runtime_fixed_layout = project_runtime_fixed_layout(
        fixed_layout,
        execution_plan.program,
        execution_plan.schedule,
        initial_task_id=INITIAL_PLACEMENT_TASK_ID,
        dynamic_task_allocations=(selected_admission.dynamic_provider_allocations()),
    )
    bridge = RuntimeBridge(memory.runtime, execution_plan.program, memory.plan_handle)
    state: MaterializedForwardState | None = None
    try:
        with timer.measure("materialization"):
            state = MaterializedForwardState(
                model,
                program.lowered,
                captured.capture,
                captured.cpu_inputs,
                bridge,
                runtime=memory.runtime,
                device_ordinal=captured.device_ordinal,
            )
        with timer.measure("physical_sealing"):
            seal_physical_budget(captured.installed, execution_plan)
        with timer.measure("callable_construction"):
            executor = ForwardExecutor(
                captured.partitioned,
                program.lowered,
                execution_plan,
                bridge,
                state,
                profiled.compiled_tasks.functions,
                captured.capture.user_output_indices,
                captured.output_tree_spec,
                fixed_layout=runtime_fixed_layout,
                memory_envelopes=selected_admission.envelopes_by_task(),
            )
        report = _forward_plan_report(
            model,
            captured,
            profiled,
            program,
            selection,
            selected_admission,
            admitted_result,
            execution_plan,
            artifact_cache=artifact_cache,
            memory=memory,
            timer=timer,
            started=started,
        )
        return PlannedForward(
            model,
            captured.signature,
            executor,
            state,
            report,
            memory.runtime,
            memory.plan_handle,
        )
    except BaseException as error:
        if state is not None:
            _rollback_forward_failure(
                memory.runtime,
                error,
                state.restore_cpu_and_unregister,
                operation="admit forward plan",
            )
        raise


def _rollback_forward_failure(
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
            f"Failed to roll back materialized forward state: {cleanup_error}"
        )
    raise error


def _forward_execution_plan(
    lowered: LoweredForwardProgram,
    selection: PressureFitResult,
    admission: PhysicalAdmission,
) -> ExecutionPlan:
    entrypoints = tuple(
        EntrypointSpec(
            task_id=item.task_id,
            entrypoint_id=f"entrypoint_{index:06d}",
            executor_id="pytorch_inductor",
            contract_digest=item.artifact.compatibility_digest,
        )
        for index, item in enumerate(lowered.entrypoints)
    )
    return selection.to_execution_plan(
        entrypoints=entrypoints,
        admission=admission,
    )


def _forward_plan_report(
    model: nn.Module,
    captured: ForwardCaptureArtifacts,
    profiled: ForwardProfileArtifacts,
    program: ForwardProgramArtifacts,
    selection: FixedLayoutSelection,
    selected_admission: SelectedAdmission,
    admitted_result: PressureFitResult,
    execution_plan: ExecutionPlan,
    *,
    artifact_cache: PlanningArtifactRepositories,
    memory: PlanMemory,
    timer: PlanningTimer,
    started: int,
) -> PlanReport:
    with timer.measure("diagnostic_inventory"):
        task_stage_map, unique_stages = forward_stage_inventory(
            program.lowered,
            execution_plan,
            program.measurements_by_profile,
            profiled.compiled_tasks.manifests,
            profiling_metadata_digest=captured.workload.digest,
        )
    report = build_forward_report(
        captured.signature.digest,
        execution_plan,
        profiled.profiles,
        tuple(timer.values),
        started,
        recomputation_cache_hit=selection.cache_hit,
        pressurefit_results=(admitted_result,),
        captured_stage_count=len(captured.partitioned.stages),
        aot_unique_stage_contracts=profiled.profiles.unique_keys,
        task_stage_map=task_stage_map,
        unique_stages=unique_stages,
        compiler_phase_timings_ns=profiled.profiler.compilation_phase_timings_ns,
        compiler_phase_timings_by_contract=(
            profiled.profiler.compilation_phase_timings_by_contract
        ),
        cache_directories=artifact_cache.store.diagnostics(),
        touched_cache_artifacts=cache_artifacts(artifact_cache.store),
        profiling_metadata=(captured.workload,),
        physical_layouts=(
            fixed_layout_diagnostic(
                "forward",
                selection,
                selected_admission,
            ),
        ),
        memory=memory,
    )
    return publish_plan_report(
        model,
        report,
        artifact_cache.store,
        started=started,
    )


def build_forward(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    memory: PlanMemory,
    partition: PartitionSpec,
    verbose: bool,
    planning_cache: PlanningCache,
    profiling_metadata: object,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
) -> PlannedForward:
    """Compose the independently callable forward-planning boundaries."""

    started = time.perf_counter_ns()
    timer = PlanningTimer(verbose=verbose)
    artifacts = open_artifact_repositories(planning_cache)
    captured = capture_forward_graph(
        model,
        example_inputs=example_inputs,
        memory=memory,
        partition=partition,
        profiling_metadata=profiling_metadata,
        artifact_cache=artifacts,
        timer=timer,
    )
    profiled = profile_forward_tasks(
        captured,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
        artifact_cache=artifacts,
        timer=timer,
    )
    program = build_forward_program(captured, profiled, memory=memory, timer=timer)
    selected = pressurefit_forward_program(
        program,
        artifact_cache=artifacts,
        timer=timer,
    )
    return admit_forward_plan(
        model,
        captured,
        profiled,
        program,
        selected,
        memory=memory,
        artifact_cache=artifacts,
        timer=timer,
        started=started,
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


__all__ = [
    "ForwardCaptureArtifacts",
    "ForwardProfileArtifacts",
    "ForwardProgramArtifacts",
    "admit_forward_plan",
    "build_forward",
    "build_forward_program",
    "capture_forward_graph",
    "pressurefit_forward_program",
    "profile_forward_tasks",
]
