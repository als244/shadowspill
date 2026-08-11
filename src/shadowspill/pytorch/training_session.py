"""Rollback-safe public accumulated-training planning session."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.ir import EntrypointSpec, PhysicalAdmission
from shadowspill.planner._cache import PressureFitCache
from shadowspill.runtime import AdmissionError
from shadowspill.simulator import SimulationConfig

from .aot import capture_training
from .compiler import CudaTaskProfiler, profile_environment
from .contracts import ObjectiveResult, PlanningError
from .fake import fake_cuda_inputs, fake_cuda_model
from .guards import capture_training_signatures
from .materialization import representative_cpu_inputs
from .optimizer import OptimizerTaskArtifact, capture_optimizer
from .partition import partition_training_capture
from .profiling import ProfileCache, profile_unique_artifacts
from .public import PlannedTrainStep, PlanReport
from .runtime_bridge import RuntimeBridge
from .session import (
    _ensure_allocator,
    _forward_report,
    _host_arena_estimate,
    _PhaseTimer,
    _reconcile_host_arena,
    _seal_physical_budget,
    _simulation_capacity,
    _workspace_reserve,
)
from .spatial_admission import replay_selected_schedule
from .training_executor import TrainingExecutor
from .training_lowering import (
    lower_partitioned_training_program,
    lower_training_storage_layout,
)
from .training_materialization import TrainingMaterializedState


def build_training(
    model: nn.Module,
    *,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    opt: Callable[[Any], torch.optim.Optimizer],
    example_inputs: Sequence[Sequence[Any]],
    device_budget: int,
    host_budget: int,
    partition: str,
) -> PlannedTrainStep:
    """Construct one fixed accumulated training program."""

    started = time.perf_counter_ns()
    timer = _PhaseTimer()
    with timer.measure("validation"):
        _validate_training_request(model, objective, opt, device_budget, host_budget)
        signatures = capture_training_signatures(example_inputs)
        cpu_inputs = tuple(
            tuple(representative_cpu_inputs(microbatch))
            for microbatch in example_inputs
        )
        host_arena = _host_arena_estimate(model, cpu_inputs, host_budget)

    with timer.measure("allocator_bootstrap"):
        installed = _ensure_allocator(
            device_budget=device_budget,
            host_arena=host_arena,
            device_ordinal=0,
        )

    with timer.measure("capture_lowering"):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_model = fake_cuda_model(model, fake_mode, device_index=0)
        with fake_mode:
            captures = tuple(
                capture_training(
                    fake_model,
                    objective,
                    fake_cuda_inputs(microbatch, fake_mode, device_index=0),
                )
                for microbatch in cpu_inputs
            )
            partitioned_captures = tuple(
                partition_training_capture(capture, partition=partition)
                for capture in captures
            )
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
                device_ordinal=0,
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
                dict(model.named_parameters()), optimizer
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

        artifact_by_digest: dict[str, OptimizerTaskArtifact] = {
            artifact.compatibility_digest: artifact
            for capture in partitioned_captures
            for stage in capture.stages
            for pair in (stage.save_pair, stage.recompute_pair)
            for artifact in (pair.forward, pair.backward)
        }
        for optimizer_task in optimizer_capture.recurrent_tasks:
            artifact_by_digest[optimizer_task.artifact.compatibility_digest] = (
                optimizer_task.artifact
            )
        if optimizer_capture.initialized_state_dict is None:
            artifact_by_digest[optimizer_capture.recurrent.compatibility_digest] = (
                optimizer_capture.recurrent
            )
        artifacts = tuple(artifact_by_digest.values())
        profiler = CudaTaskProfiler(installed.library, device_ordinal=0)
        with timer.measure("structural_profiling"):
            profiles = profile_unique_artifacts(
                artifacts,
                environment=profile_environment(
                    device_ordinal=0, provider_id="shadowspill.cuda_slab"
                ),
                measure=profiler.measure,
                cache=ProfileCache(),
            )
        with timer.measure("compilation"):
            functions = profiler.take_functions(artifacts)
            installed.library.shadowspill_pytorch_allocator_wait_idle()

        with timer.measure("program_lowering"):
            measurements = {
                artifact.compatibility_digest: measurement
                for artifact, measurement in zip(
                    artifacts, profiles.measurements, strict=True
                )
            }
            initial_lowered = lower_partitioned_training_program(
                fake_model,
                partitioned_captures,
                measurements,
                optimizer_capture,
                optimizer_phase="initial",
            )
            recurrent_lowered = lower_partitioned_training_program(
                fake_model,
                partitioned_captures,
                measurements,
                optimizer_capture,
                optimizer_phase="recurrent",
            )
            _verify_provisional_layout(layout, recurrent_lowered)
            _verify_optimizer_phase_identity(initial_lowered, recurrent_lowered)
            workspace_reserve = _workspace_reserve(profiles.measurements)
            simulation_config = SimulationConfig.single_device(
                "cuda_0",
                device_capacity_bytes=_simulation_capacity(
                    int(installed.admission.slab_bytes),
                    workspace_reserve,
                    profiles.measurements,
                ),
                host_capacity_bytes=host_budget,
                h2d_bandwidth_bytes_per_second=24 << 30,
                d2h_bandwidth_bytes_per_second=24 << 30,
                h2d_latency_ns=5_000,
                d2h_latency_ns=5_000,
            )
        with timer.measure("pressurefit_simulation"):
            selection_cache = PressureFitCache()
            recurrent_cached = selection_cache.resolve(
                recurrent_lowered.program,
                initial_residency=recurrent_lowered.initial_residency,
                final_residency=recurrent_lowered.final_residency,
                config=simulation_config,
            )
            recurrent_selected = recurrent_cached.result
            needs_initial_plan = any(
                item.created_on_first_step for item in initial_lowered.optimizer_objects
            )
            initial_cached = (
                selection_cache.resolve(
                    initial_lowered.program,
                    initial_residency=initial_lowered.initial_residency,
                    final_residency=initial_lowered.final_residency,
                    config=simulation_config,
                )
                if needs_initial_plan
                else None
            )
            initial_selected = None if initial_cached is None else initial_cached.result
        with timer.measure("host_admission"):
            _reconcile_host_arena(
                installed,
                predicted_host_peak=max(
                    recurrent_selected.simulation.host_peak_bytes,
                    0
                    if initial_selected is None
                    else initial_selected.simulation.host_peak_bytes,
                ),
                host_budget=host_budget,
            )
        with timer.measure("slab_admission"):
            try:
                replays = [
                    replay_selected_schedule(
                        recurrent_selected,
                        measurements,
                        slab_bytes=int(installed.admission.slab_bytes),
                    )
                ]
                if initial_selected is not None:
                    replays.append(
                        replay_selected_schedule(
                            initial_selected,
                            measurements,
                            slab_bytes=int(installed.admission.slab_bytes),
                        )
                    )
            except AdmissionError as exc:
                raise PlanningError(f"slab spatial admission failed: {exc}") from exc
            predicted_fragmentation = max(
                item.peak_fragmentation_bytes for item in replays
            )
        admission = PhysicalAdmission(
            device_budget_bytes=device_budget,
            host_budget_bytes=host_budget,
            context_bytes=int(installed.admission.context_bytes),
            provider_headroom_bytes=int(installed.admission.provider_headroom_bytes),
            slab_bytes=int(installed.admission.slab_bytes),
            workspace_reserve_bytes=workspace_reserve,
            host_reservation_bytes=int(installed.admission.host_arena_bytes),
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
            _seal_physical_budget(installed)
        with timer.measure("callable_construction"):
            executor = TrainingExecutor(
                None if initial_plan is None else (initial_lowered, initial_plan),
                (recurrent_lowered, recurrent_plan),
                final_bridge,
                state,
                functions,
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
                    + (
                        0
                        if initial_cached is None
                        else int(not initial_cached.cache_hit)
                    )
                ),
            )
            return PlannedTrainStep(
                model, signatures, executor, state, optimizer, report
            )
    except BaseException:
        if state is not None:
            for parameter in model.parameters():
                parameter.grad = None
            state.restore_cpu_and_unregister()
        raise


def _validate_training_request(
    model: nn.Module,
    objective: object,
    opt: object,
    device_budget: int,
    host_budget: int,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not callable(objective):
        raise TypeError("objective must be callable")
    if not callable(opt):
        raise TypeError("opt must be an optimizer factory")
    if isinstance(device_budget, bool) or not isinstance(device_budget, int):
        raise TypeError("device_budget must be an integer byte count")
    if isinstance(host_budget, bool) or not isinstance(host_budget, int):
        raise TypeError("host_budget must be an integer byte count")
    if device_budget <= 512 << 20:
        raise PlanningError("device_budget must exceed provider headroom")
    if host_budget <= 0:
        raise PlanningError("host_budget must be positive")
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
) -> PlanReport:
    identity = {
        "mode": "training",
        "signatures": signature_digests,
        "artifacts": [item.abi_digest for item in execution_plan.entrypoints],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # The report construction is identical to forward apart from capture identity.
    report = _forward_report(digest, execution_plan, profiles, timings, started)
    return PlanReport(
        mode="training",
        capture_identity=digest,
        execution_plan=report.execution_plan,
        task_profiles=report.task_profiles,
        transfer_actions=report.transfer_actions,
        transfer_bytes_to_host=report.transfer_bytes_to_host,
        transfer_bytes_to_device=report.transfer_bytes_to_device,
        profile_unique_keys=report.profile_unique_keys,
        profile_cache_hits=report.profile_cache_hits,
        profile_cache_misses=report.profile_cache_misses,
        profiling_provenance=report.profiling_provenance,
        phase_timings_ns=report.phase_timings_ns,
        initial_execution_plan=initial_execution_plan,
        recomputation_cache_hits=recomputation_cache_hits,
        recomputation_cache_misses=recomputation_cache_misses,
    )


__all__ = ["build_training"]
