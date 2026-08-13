"""Rollback-safe public accumulated-training planning session."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.ir import EntrypointSpec, PhysicalAdmission
from shadowspill.runtime import AdmissionError
from shadowspill.simulator import SimulationConfig

from ._plan_diagnostics import training_stage_inventory
from ._planning_artifacts import planning_artifacts
from ._planning_cache import PlanningCache
from ._profiling_metadata import training_profiling_metadata
from .aot import capture_training_objective
from .capture import GraphArtifact
from .compiler import (
    CudaTaskProfiler,
    profile_environment,
    resolve_task_manifests,
    validate_compiled_profile,
)
from .contracts import ObjectiveResult, PlanningError
from .fake import fake_cuda_inputs, fake_cuda_model
from .guards import capture_training_signatures
from .materialization import representative_cpu_inputs
from .optimizer import OptimizerTaskArtifact, capture_optimizer
from .partition import partition_training_capture, training_parameter_stage_owners
from .profiling import TaskMeasurement, profile_unique_artifacts
from .public import (
    PlanDiagnostics,
    PlannedTrainStep,
    PlanReport,
    PlanTaskStage,
    PlanUniqueStage,
)
from .runtime import PlanMemory
from .runtime_bridge import RuntimeBridge
from .session import (
    _finalize_plan_report,
    _forward_report,
    _PhaseTimer,
    _public_cache_artifacts,
    _reconcile_spill_pool,
    _seal_physical_budget,
    _simulation_capacity,
    _spill_pool_estimate,
    _workspace_reserve,
)
from .spatial_admission import (
    output_bindings_for_entrypoints,
    replay_selected_schedule,
)
from .training_executor import TrainingExecutor
from .training_lowering import (
    ProfileMeasurementKey,
    TrainingLoweringCache,
    lower_partitioned_training_program,
    lower_training_storage_layout,
)
from .training_materialization import (
    TrainingMaterializedState,
    representative_training_arguments,
)


def build_training(
    model: nn.Module,
    *,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    opt: Callable[[Any], torch.optim.Optimizer],
    example_inputs: Sequence[Sequence[Any]],
    memory: PlanMemory,
    partition: str,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    verbose: bool,
    planning_cache: PlanningCache,
    profiling_metadata: Sequence[object] | None,
) -> PlannedTrainStep:
    """Construct one fixed accumulated training program."""

    started = time.perf_counter_ns()
    timer = _PhaseTimer(verbose=verbose)
    artifact_store = planning_artifacts(planning_cache)
    workloads = training_profiling_metadata(
        profiling_metadata,
        microbatch_count=len(example_inputs),
    )
    with timer.measure("validation"):
        _validate_training_request(
            model,
            objective,
            opt,
            memory.execution_budget,
            memory.spill_budget,
        )
        signatures = capture_training_signatures(example_inputs)
        cpu_inputs = tuple(
            tuple(representative_cpu_inputs(microbatch))
            for microbatch in example_inputs
        )
        _spill_pool_estimate(model, cpu_inputs, memory.spill_budget)

    with timer.measure("runtime_binding"):
        installed = memory.installed
        device_ordinal = memory.execution_device

    with timer.measure("capture_lowering"):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_model = fake_cuda_model(model, fake_mode, device_index=device_ordinal)
        with fake_mode, timer.measure("objective_export"):
            captures = tuple(
                capture_training_objective(
                    fake_model,
                    objective,
                    fake_cuda_inputs(
                        microbatch, fake_mode, device_index=device_ordinal
                    ),
                )
                for microbatch in cpu_inputs
            )
        with timer.measure("export_archival"):
            for position, objective_capture in enumerate(captures):
                artifact_store.archive_export(
                    objective_capture.exported,
                    mode="training_objective",
                    position=position,
                )
        representative_roots = tuple(
            representative_training_arguments(
                objective_capture,
                model,
                microbatch,
            )
            for objective_capture, microbatch in zip(
                captures,
                cpu_inputs,
                strict=True,
            )
        )
        with fake_mode, timer.measure("stage_partition_aot"):
            graph_pair_cache = artifact_store.graph_pairs
            partitioned_captures = tuple(
                partition_training_capture(
                    capture,
                    partition=partition,
                    graph_pair_cache=graph_pair_cache,
                    representative_root_inputs=root_inputs,
                )
                for capture, root_inputs in zip(
                    captures,
                    representative_roots,
                    strict=True,
                )
            )
        with timer.measure("storage_layout_lowering"):
            layout = lower_training_storage_layout(fake_model, captures)

    provisional_bridge = RuntimeBridge(installed.library, layout.program)
    state: TrainingMaterializedState | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        with timer.measure("model_materialization"):
            state = TrainingMaterializedState(
                model,
                layout,
                captures,
                cpu_inputs,
                provisional_bridge,
                device_ordinal=device_ordinal,
            )
        with timer.measure("optimizer_capture"):
            optimizer = opt(model.parameters())
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise PlanningError("optimizer factory must return Optimizer")
            state.restore_model_cpu_for_optimizer_capture()
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.zeros_like(parameter)
            optimizer_capture = capture_optimizer(
                dict(model.named_parameters()),
                optimizer,
                parameter_stage_owners=training_parameter_stage_owners(
                    partitioned_captures,
                    dict(model.named_parameters()),
                ),
            )
            if optimizer_capture.initialized_state_dict is not None:
                optimizer.load_state_dict(optimizer_capture.initialized_state_dict)
            for parameter in model.parameters():
                parameter.grad = None
            state.restore_cuda_placeholders_after_optimizer_capture()
            if optimizer_capture.recurrent is None:
                raise PlanningError(
                    "the optimizer state/update cannot be bounded: "
                    f"{optimizer_capture.opaque_reason}"
                )

        artifact_by_digest: dict[str, OptimizerTaskArtifact] = {}
        profile_artifact_by_key: dict[
            tuple[str, str | None], OptimizerTaskArtifact
        ] = {}
        for position, partitioned_capture in enumerate(partitioned_captures):
            metadata_digest = workloads[position].digest
            for stage in partitioned_capture.stages:
                for pair in (stage.save_pair, stage.recompute_pair):
                    for artifact in (pair.forward, pair.backward):
                        artifact_by_digest.setdefault(
                            artifact.compatibility_digest,
                            artifact,
                        )
                        profile_artifact_by_key.setdefault(
                            (artifact.compatibility_digest, metadata_digest),
                            artifact,
                        )
        for optimizer_task in optimizer_capture.recurrent_tasks:
            artifact_by_digest.setdefault(
                optimizer_task.artifact.compatibility_digest,
                optimizer_task.artifact,
            )
            profile_artifact_by_key.setdefault(
                (optimizer_task.artifact.compatibility_digest, None),
                optimizer_task.artifact,
            )
        if optimizer_capture.initialized_state_dict is None:
            artifact_by_digest.setdefault(
                optimizer_capture.recurrent.compatibility_digest,
                optimizer_capture.recurrent,
            )
            profile_artifact_by_key.setdefault(
                (optimizer_capture.recurrent.compatibility_digest, None),
                optimizer_capture.recurrent,
            )
        artifacts = tuple(artifact_by_digest.values())
        profile_keys = tuple(profile_artifact_by_key)
        profile_artifacts = tuple(profile_artifact_by_key.values())
        profile_metadata_digests = tuple(item[1] for item in profile_keys)
        optimizer_artifact_count = sum(
            not isinstance(item, GraphArtifact) or item.kind == "optimizer"
            for item in artifacts
        )
        timer.progress(
            "structural artifact inventory: "
            f"graph={len(artifacts) - optimizer_artifact_count}, "
            f"optimizer={optimizer_artifact_count}, "
            f"unique={len(artifacts)}, "
            f"profile_variants={len(profile_artifacts)}, "
            f"optimizer_tasks={len(optimizer_capture.recurrent_tasks)}"
        )
        profiler = CudaTaskProfiler(installed.library, device_ordinal=device_ordinal)
        environment = profile_environment(
            device_ordinal=device_ordinal,
            provider_id="shadowspill.device_pool",
            implementation_revision=planning_cache.implementation_revision,
        )
        with timer.measure("compiler_manifest"):
            resolved_manifests = resolve_task_manifests(
                artifacts,
                environment=environment,
                profile_cache=artifact_store.profiles,
                profiler=profiler,
                progress=lambda index, total, state, digest: timer.progress(
                    f"compiled manifest {index}/{total} {state}: {digest[:12]}"
                ),
            )
            timer.progress(
                "compiled manifest cache: "
                f"hits={resolved_manifests.cache_hits}, "
                f"misses={resolved_manifests.cache_misses}"
            )
        with timer.measure("structural_profiling"):
            profiles = profile_unique_artifacts(
                profile_artifacts,
                environment=environment,
                measure=profiler.measure,
                cache=artifact_store.profiles,
                validate=lambda artifact, measurement: validate_compiled_profile(
                    artifact,
                    measurement,
                    resolved_manifests.manifests,
                ),
                progress=lambda index, total, state, digest: timer.progress(
                    f"structural profile {index}/{total} {state}: {digest[:12]}"
                ),
                profiling_metadata_digests=profile_metadata_digests,
            )

        with timer.measure("program_lowering"):
            measurements: dict[ProfileMeasurementKey, TaskMeasurement] = {
                key: measurement
                for key, measurement in zip(
                    profile_keys, profiles.measurements, strict=True
                )
            }
            measurements_by_profile = dict(
                zip(profiles.key_digests, profiles.measurements, strict=True)
            )
            profile_compatibility_digests = {
                key: digest
                for key, digest in zip(profile_keys, profiles.key_digests, strict=True)
            }
            lowering_cache = TrainingLoweringCache()
            initial_lowered = lower_partitioned_training_program(
                fake_model,
                partitioned_captures,
                measurements,
                optimizer_capture,
                storage_contracts={
                    digest: manifest.storage_contract
                    for digest, manifest in resolved_manifests.manifests.items()
                },
                compiled_root_allocations={
                    digest: manifest.root_allocations
                    for digest, manifest in resolved_manifests.manifests.items()
                },
                optimizer_phase="initial",
                optimizer_ordering=optimizer_ordering,
                lowering_cache=lowering_cache,
                profiling_metadata_digests=tuple(item.digest for item in workloads),
                profile_compatibility_digests=profile_compatibility_digests,
            )
            recurrent_lowered = lower_partitioned_training_program(
                fake_model,
                partitioned_captures,
                measurements,
                optimizer_capture,
                storage_contracts={
                    digest: manifest.storage_contract
                    for digest, manifest in resolved_manifests.manifests.items()
                },
                compiled_root_allocations={
                    digest: manifest.root_allocations
                    for digest, manifest in resolved_manifests.manifests.items()
                },
                optimizer_phase="recurrent",
                optimizer_ordering=optimizer_ordering,
                lowering_cache=lowering_cache,
                profiling_metadata_digests=tuple(item.digest for item in workloads),
                profile_compatibility_digests=profile_compatibility_digests,
            )
            _verify_provisional_layout(layout, recurrent_lowered)
            _verify_optimizer_phase_identity(initial_lowered, recurrent_lowered)
            largest_profile = max(
                recurrent_lowered.program.profiles,
                key=lambda item: item.workspace_bytes,
            )
            timer.progress(
                "recurrent Program inventory: "
                f"tasks={len(recurrent_lowered.program.tasks)}, "
                f"objects={len(recurrent_lowered.program.objects)}, "
                f"aliases={len(recurrent_lowered.program.alias_groups)}, "
                "recomputation_groups="
                f"{len(recurrent_lowered.program.recomputation_groups)}, "
                f"largest_workspace={largest_profile.workspace_bytes} "
                f"({largest_profile.profile_id})"
            )
            workspace_reserve = _workspace_reserve(profiles.measurements)
            simulation_config = SimulationConfig.single_device(
                "cuda_0",
                device_capacity_bytes=_simulation_capacity(
                    memory.execution_budget,
                    workspace_reserve,
                    profiles.measurements,
                    fixed_slab_bytes=profiles.fixed_slab_bytes,
                ),
                host_capacity_bytes=memory.spill_budget,
                fetch_bandwidth_bytes_per_second=memory.transfers.route(
                    memory.spill.name, memory.execution.name
                ).bandwidth_bytes_per_second,
                evict_bandwidth_bytes_per_second=memory.transfers.route(
                    memory.execution.name, memory.spill.name
                ).bandwidth_bytes_per_second,
                fetch_latency_ns=memory.transfers.route(
                    memory.spill.name, memory.execution.name
                ).latency_nanoseconds,
                evict_latency_ns=memory.transfers.route(
                    memory.execution.name, memory.spill.name
                ).latency_nanoseconds,
            )
        with timer.measure("pressurefit_simulation"):
            recurrent_cached = artifact_store.select(
                recurrent_lowered.program,
                initial_residency=recurrent_lowered.initial_residency,
                final_residency=recurrent_lowered.final_residency,
                config=simulation_config,
                progress=timer.progress,
            )
            recurrent_selected = recurrent_cached.result
            needs_initial_plan = any(
                item.created_on_first_step for item in initial_lowered.optimizer_objects
            )
            initial_cached = (
                artifact_store.select(
                    initial_lowered.program,
                    initial_residency=initial_lowered.initial_residency,
                    final_residency=initial_lowered.final_residency,
                    config=simulation_config,
                    progress=timer.progress,
                )
                if needs_initial_plan
                else None
            )
            initial_selected = None if initial_cached is None else initial_cached.result
        required_digests = _selected_artifact_digests(
            recurrent_lowered,
            recurrent_selected,
        )
        if initial_selected is not None:
            required_digests.update(
                _selected_artifact_digests(initial_lowered, initial_selected)
            )
        required_artifacts = tuple(
            artifact
            for artifact in artifacts
            if artifact.compatibility_digest in required_digests
        )
        with timer.measure("compilation"):
            compiled_tasks = profiler.take_compiled_tasks(
                required_artifacts,
                progress=lambda index, total, state, digest: timer.progress(
                    f"selected entrypoint {index}/{total} {state}: {digest[:12]}"
                ),
            )
            _verify_compiled_manifest_identity(
                resolved_manifests.manifests,
                compiled_tasks.manifests,
            )
            profiler.discard_compiled_tasks()
            installed.library.shadowspill_pytorch_allocator_wait_idle()
        timer.attribute_compilation_and_profiling(profiler)
        with timer.measure("host_admission"):
            _reconcile_spill_pool(
                installed,
                predicted_host_peak=max(
                    recurrent_selected.simulation.host_peak_bytes,
                    0
                    if initial_selected is None
                    else initial_selected.simulation.host_peak_bytes,
                ),
                host_budget=memory.spill_budget,
            )
        with timer.measure("slab_admission"):
            try:
                replays = [
                    replay_selected_schedule(
                        recurrent_selected,
                        measurements_by_profile,
                        execution_pool_bytes=(
                            memory.execution_budget - profiles.fixed_slab_bytes
                        ),
                        output_bindings=output_bindings_for_entrypoints(
                            recurrent_selected.program.selected_tasks(
                                recurrent_selected.selections
                            ),
                            recurrent_lowered.entrypoints,
                            {
                                item.object_id: item.alias_group_id
                                for item in recurrent_selected.program.objects
                            },
                        ),
                    )
                ]
                if initial_selected is not None:
                    replays.append(
                        replay_selected_schedule(
                            initial_selected,
                            measurements_by_profile,
                            execution_pool_bytes=(
                                memory.execution_budget - profiles.fixed_slab_bytes
                            ),
                            output_bindings=output_bindings_for_entrypoints(
                                initial_selected.program.selected_tasks(
                                    initial_selected.selections
                                ),
                                initial_lowered.entrypoints,
                                {
                                    item.object_id: item.alias_group_id
                                    for item in initial_selected.program.objects
                                },
                            ),
                        )
                    )
            except AdmissionError as exc:
                raise PlanningError(f"slab spatial admission failed: {exc}") from exc
            predicted_fragmentation = max(
                item.peak_fragmentation_bytes for item in replays
            )
        admission = PhysicalAdmission(
            device_budget_bytes=(
                memory.execution.physical_capacity or memory.execution_budget
            ),
            host_budget_bytes=memory.spill_budget,
            context_bytes=int(installed.admission.context_bytes),
            provider_headroom_bytes=int(installed.admission.provider_headroom_bytes),
            slab_bytes=memory.execution_budget,
            workspace_reserve_bytes=workspace_reserve,
            host_reservation_bytes=int(installed.admission.spill_pool_bytes),
            predicted_fragmentation_bytes=predicted_fragmentation,
        )
        recurrent_plan = _execution_plan(
            recurrent_lowered,
            recurrent_selected,
            optimizer_capture.optimizer_type,
            admission,
        )
        initial_plan = (
            _execution_plan(
                initial_lowered,
                initial_selected,
                optimizer_capture.optimizer_type,
                admission,
            )
            if initial_selected is not None
            else None
        )
        final_bridge = RuntimeBridge(installed.library, recurrent_plan.program)
        with timer.measure("plan_adoption"):
            state.adopt_execution_plan(
                final_bridge,
                recurrent_lowered,
                optimizer=optimizer,
            )
        with timer.measure("physical_sealing"):
            _seal_physical_budget(installed, recurrent_plan)
        with timer.measure("diagnostic_inventory"):
            task_stage_map, unique_stages = training_stage_inventory(
                partitioned_captures,
                recurrent_lowered,
                recurrent_plan,
                measurements,
                resolved_manifests.manifests,
                profiling_metadata_digests=tuple(item.digest for item in workloads),
            )
        with timer.measure("callable_construction"):
            executor = TrainingExecutor(
                None if initial_plan is None else (initial_lowered, initial_plan),
                (recurrent_lowered, recurrent_plan),
                final_bridge,
                state,
                compiled_tasks.functions,
                optimizer,
                optimizer_state_preinitialized=(
                    optimizer_capture.initialized_state_dict is not None
                ),
                optimizer_state_was_lazy=bool(optimizer_capture.created_state_names),
            )
        report = _training_report(
            tuple(signature.digest for signature in signatures),
            recurrent_plan,
            profiles,
            tuple(timer.values),
            started,
            initial_execution_plan=initial_plan,
            recomputation_cache_hits=(
                int(recurrent_cached.cache_hit)
                + (0 if initial_cached is None else int(initial_cached.cache_hit))
            ),
            recomputation_cache_misses=(
                int(not recurrent_cached.cache_hit)
                + (0 if initial_cached is None else int(not initial_cached.cache_hit))
            ),
            captured_stage_count=sum(
                len(capture.stages) for capture in partitioned_captures
            ),
            aot_unique_stage_abis=graph_pair_cache.unique_keys,
            aot_graph_pair_cache_hits=graph_pair_cache.hits,
            aot_graph_pair_cache_misses=graph_pair_cache.misses,
            pressurefit_results=tuple(
                item
                for item in (initial_selected, recurrent_selected)
                if item is not None
            ),
            task_stage_map=task_stage_map,
            unique_stages=unique_stages,
            compiler_phase_timings_ns=profiler.compilation_phase_timings_ns,
            compiler_phase_timings_by_abi=(profiler.compilation_phase_timings_by_abi),
            cache_directories=planning_cache.diagnostics(),
            cache_artifacts=_public_cache_artifacts(planning_cache),
            profiling_metadata=workloads,
            optimizer_ordering=optimizer_ordering,
            memory=memory,
        )
        report = _finalize_plan_report(
            model,
            report,
            planning_cache,
            started=started,
        )
        return PlannedTrainStep(
            model,
            signatures,
            executor,
            state,
            optimizer,
            report,
            memory.runtime,
        )
    except BaseException:
        if state is not None:
            for parameter in model.parameters():
                parameter.grad = None
            state.restore_cpu_and_unregister()
        raise


def _selected_artifact_digests(lowered: Any, selected: Any) -> set[str]:
    selected_task_ids = {
        task.task_id for task in lowered.program.selected_tasks(selected.selections)
    }
    return {
        entrypoint.artifact.compatibility_digest
        for entrypoint in lowered.entrypoints
        if entrypoint.task_id in selected_task_ids and entrypoint.artifact is not None
    }


def _verify_compiled_manifest_identity(
    planned: dict[str, Any],
    executable: dict[str, Any],
) -> None:
    for digest, manifest in executable.items():
        expected = planned.get(digest)
        if expected is None or (
            expected.compatibility_digest != manifest.compatibility_digest
        ):
            raise PlanningError(
                "selected compiled entrypoint changed its storage ABI: "
                f"artifact={digest}"
            )


def _validate_training_request(
    model: nn.Module,
    objective: object,
    opt: object,
    execution_budget: int,
    spill_budget: int,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not callable(objective):
        raise TypeError("objective must be callable")
    if not callable(opt):
        raise TypeError("opt must be an optimizer factory")
    if isinstance(execution_budget, bool) or not isinstance(execution_budget, int):
        raise TypeError("execution_budget must be an integer byte count")
    if isinstance(spill_budget, bool) or not isinstance(spill_budget, int):
        raise TypeError("spill_budget must be an integer byte count")
    if execution_budget <= 0:
        raise PlanningError("execution_budget must be positive")
    if spill_budget <= 0:
        raise PlanningError("spill_budget must be positive")
    for name, tensor in tuple(model.named_parameters()) + tuple(model.named_buffers()):
        if tensor.device.type != "cpu":
            raise PlanningError(f"registered tensor {name!r} must be CPU resident")


def _verify_provisional_layout(layout: Any, lowered: Any) -> None:
    expected = {item.object_id: item.alias_group_id for item in layout.program.objects}
    actual = {item.object_id: item.alias_group_id for item in lowered.program.objects}
    if any(
        actual.get(object_id) != alias_id for object_id, alias_id in expected.items()
    ):
        raise PlanningError(
            "training storage identities changed after optimizer capture"
        )


def _verify_optimizer_phase_identity(initial: Any, recurrent: Any) -> None:
    if initial.program.alias_groups != recurrent.program.alias_groups or (
        initial.program.objects != recurrent.program.objects
    ):
        raise PlanningError("optimizer phases changed storage identities")


def _execution_plan(
    lowered: Any,
    selected: Any,
    optimizer_type: str,
    admission: PhysicalAdmission,
) -> Any:
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


def _training_report(
    signature_digests: tuple[str, ...],
    execution_plan: Any,
    profiles: Any,
    timings: tuple[tuple[str, int], ...],
    started: int,
    *,
    initial_execution_plan: Any = None,
    recomputation_cache_hits: int = 0,
    recomputation_cache_misses: int = 0,
    captured_stage_count: int = 0,
    aot_unique_stage_abis: int = 0,
    aot_graph_pair_cache_hits: int = 0,
    aot_graph_pair_cache_misses: int = 0,
    pressurefit_results: tuple[Any, ...] = (),
    task_stage_map: tuple[PlanTaskStage, ...] = (),
    unique_stages: tuple[PlanUniqueStage, ...] = (),
    compiler_phase_timings_ns: tuple[tuple[str, int], ...] = (),
    compiler_phase_timings_by_abi: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ] = (),
    cache_directories: tuple[tuple[str, str], ...] = (),
    cache_artifacts: tuple[Any, ...] = (),
    profiling_metadata: tuple[Any, ...] = (),
    optimizer_ordering: str,
    memory: PlanMemory,
) -> PlanReport:
    identity = {
        "mode": "training",
        "signatures": signature_digests,
        "artifacts": [item.abi_digest for item in execution_plan.entrypoints],
        "optimizer_ordering": optimizer_ordering,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # The report construction is identical to forward apart from capture identity.
    report = _forward_report(
        digest,
        execution_plan,
        profiles,
        timings,
        started,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        compiler_phase_timings_ns=compiler_phase_timings_ns,
        compiler_phase_timings_by_abi=compiler_phase_timings_by_abi,
        cache_directories=cache_directories,
        cache_artifacts=cache_artifacts,
        profiling_metadata=profiling_metadata,
        memory=memory,
    )
    base_diagnostics = report.diagnostics
    diagnostics = PlanDiagnostics(
        phases=base_diagnostics.phases,
        total_wall_time_ns=base_diagnostics.total_wall_time_ns,
        unattributed_overhead_ns=base_diagnostics.unattributed_overhead_ns,
        profile_unique_keys=base_diagnostics.profile_unique_keys,
        profile_cache_hits=base_diagnostics.profile_cache_hits,
        profile_cache_misses=base_diagnostics.profile_cache_misses,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        recomputation_cache_hits=recomputation_cache_hits,
        recomputation_cache_misses=recomputation_cache_misses,
        task_stage_map=task_stage_map,
        unique_stages=unique_stages,
        compiler_phase_timings_ns=compiler_phase_timings_ns,
        compiler_profiles=base_diagnostics.compiler_profiles,
        cache_directories=cache_directories,
        cache_artifacts=base_diagnostics.cache_artifacts,
        profiling_metadata=base_diagnostics.profiling_metadata,
    )
    return PlanReport(
        mode="training",
        capture_identity=digest,
        execution_plan=report.execution_plan,
        task_profiles=report.task_profiles,
        transfer_actions=report.transfer_actions,
        transfer_bytes_evicted=report.transfer_bytes_evicted,
        transfer_bytes_fetched=report.transfer_bytes_fetched,
        profile_unique_keys=report.profile_unique_keys,
        profile_cache_hits=report.profile_cache_hits,
        profile_cache_misses=report.profile_cache_misses,
        profiling_provenance=report.profiling_provenance,
        phase_timings_ns=report.phase_timings_ns,
        initial_execution_plan=initial_execution_plan,
        recomputation_cache_hits=recomputation_cache_hits,
        recomputation_cache_misses=recomputation_cache_misses,
        fixed_slab_bytes=report.fixed_slab_bytes,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        pressurefit_results=pressurefit_results,
        diagnostics=diagnostics,
        execution_pool=report.execution_pool,
        spill_pool=report.spill_pool,
        execution_budget_bytes=report.execution_budget_bytes,
        spill_budget_bytes=report.spill_budget_bytes,
        execution_device=report.execution_device,
        transfer_capabilities=report.transfer_capabilities,
        optimizer_ordering=optimizer_ordering,
    )


__all__ = ["build_training"]
