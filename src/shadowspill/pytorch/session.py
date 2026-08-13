"""Rollback-safe construction shared by ShadowSpill PyTorch entrypoints."""

from __future__ import annotations

import ctypes
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import TreeSpec, tree_flatten

from shadowspill.ir import EntrypointSpec, MemoryActionKind, PhysicalAdmission
from shadowspill.planner._cache import CachedPressureFitResult
from shadowspill.runtime import AdmissionError, SlabReplay, workspace_reserve_bytes
from shadowspill.simulator import SimulationConfig

from ._abi import AdapterStatistics
from ._allocator import InstalledAllocator
from ._plan_diagnostics import forward_stage_inventory
from ._planning_artifacts import PlanningArtifacts, planning_artifacts
from ._planning_cache import PlanningCache
from ._profiling_metadata import ProfilingMetadata, canonicalize_profiling_metadata
from .aot import ExportCapture, capture_forward
from .capture import GraphArtifact
from .compiler import (
    CompiledTaskSet,
    CudaTaskProfiler,
    ResolvedTaskManifests,
    profile_environment,
    resolve_task_manifests,
    validate_compiled_profile,
)
from .contracts import PlanningError
from .executor import ForwardExecutor
from .fake import fake_cuda_inputs, fake_cuda_model
from .guards import InputSignature, capture_input_signature
from .lowering import LoweredForwardProgram, lower_forward_program
from .materialization import (
    MaterializedForwardState,
    flat_runtime_arguments,
    representative_cpu_inputs,
)
from .partition import PartitionedExport, capture_forward_stages, partition_export
from .profiling import ProfilingResult, TaskMeasurement, profile_unique_artifacts
from .public import (
    PlanCacheArtifact,
    PlanCompilerProfile,
    PlanDiagnostics,
    PlannedForward,
    PlanPhaseTiming,
    PlanProfilingMetadata,
    PlanReport,
    PlanTaskStage,
    PlanUniqueStage,
)
from .runtime import PlanMemory
from .runtime_bridge import RuntimeBridge
from .spatial_admission import (
    output_bindings_for_entrypoints,
    replay_selected_schedule,
)

_MIB = 1 << 20
_HOST_LEEWAY_MINIMUM = 256 * _MIB
_HOST_ALIGNMENT = 64 << 10


class _PhaseTimer:
    def __init__(self, *, verbose: bool) -> None:
        self.values: list[tuple[str, int]] = []
        self._verbose = verbose
        self._depth = 0
        self._started = time.perf_counter_ns()

    def measure(self, name: str) -> _MeasuredPhase:
        return _MeasuredPhase(self, name)

    def progress(self, message: str) -> None:
        if not self._verbose:
            return
        elapsed = (time.perf_counter_ns() - self._started) / 1e9
        indentation = "  " * self._depth
        print(
            f"[shadowspill.plan +{elapsed:8.3f}s] {indentation}{message}",
            flush=True,
        )

    def attribute_compilation_and_profiling(self, profiler: CudaTaskProfiler) -> None:
        """Replace compiler/profile intervals with disjoint work classes."""

        names = {"compiler_manifest", "structural_profiling", "compilation"}
        indexed = [
            (index, name, duration)
            for index, (name, duration) in enumerate(self.values)
            if name in names
        ]
        if {name for _index, name, _duration in indexed} != names:
            raise RuntimeError("planning profile intervals are incomplete")
        combined = sum(duration for _index, _name, duration in indexed)
        compilation = profiler.compilation_wall_time_ns
        profiling = profiler.profiling_wall_time_ns
        cached_warmup = profiler.entrypoint_warmup_wall_time_ns
        measured = compilation + profiling + cached_warmup
        if measured > combined:
            raise RuntimeError(
                "compiler/profile subphase clocks exceed their enclosing intervals"
            )
        replacement = [
            ("compiled_entrypoint_construction", compilation),
            ("unique_stage_warmup_profiling", profiling),
            ("cached_entrypoint_warmup", cached_warmup),
            ("profile_cache_and_entrypoint_orchestration", combined - measured),
        ]
        first = min(index for index, _name, _duration in indexed)
        retained = [item for item in self.values if item[0] not in names]
        self.values = retained[:first] + replacement + retained[first:]


class _MeasuredPhase:
    def __init__(self, timer: _PhaseTimer, name: str) -> None:
        self._timer = timer
        self._name = name
        self._start = 0

    def __enter__(self) -> None:
        self._depth = self._timer._depth
        self._timer.progress(f"{self._name}: started")
        self._timer._depth += 1
        self._start = time.perf_counter_ns()

    def __exit__(self, *exception: object) -> None:
        duration = time.perf_counter_ns() - self._start
        self._timer.values.append((self._name, duration))
        self._timer._depth = self._depth
        outcome = "failed" if exception[0] is not None else "finished"
        self._timer.progress(f"{self._name}: {outcome} in {duration / 1e9:.3f}s")


def build_forward(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    memory: PlanMemory,
    partition: str,
    verbose: bool,
    planning_cache: PlanningCache,
    profiling_metadata: object,
) -> PlannedForward:
    """Construct a planned forward callable without mutating arithmetic."""

    return _ForwardPlanningSession(
        model,
        example_inputs=example_inputs,
        memory=memory,
        partition=partition,
        verbose=verbose,
        artifacts=planning_artifacts(planning_cache),
        profiling_metadata=profiling_metadata,
    ).run()


@dataclass(frozen=True, slots=True)
class _ForwardRequest:
    signature: InputSignature
    cpu_inputs: tuple[object, ...]
    workload: ProfilingMetadata
    installed: InstalledAllocator
    device_ordinal: int


@dataclass(frozen=True, slots=True)
class _ForwardCaptureArtifacts:
    fake_model: nn.Module
    capture: ExportCapture
    partitioned: PartitionedExport
    task_artifacts: tuple[GraphArtifact, ...]
    output_tree_spec: TreeSpec


@dataclass(frozen=True, slots=True)
class _ForwardProfileArtifacts:
    profiler: CudaTaskProfiler
    manifests: ResolvedTaskManifests
    profiles: ProfilingResult
    compiled_tasks: CompiledTaskSet


@dataclass(frozen=True, slots=True)
class _ForwardProgramArtifacts:
    lowered: LoweredForwardProgram
    measurements: dict[str, TaskMeasurement]
    measurements_by_profile: dict[str, TaskMeasurement]
    workspace_reserve: int
    simulation_config: SimulationConfig


class _ForwardPlanningSession:
    """Rollback-safe orchestration for one forward planning request."""

    def __init__(
        self,
        model: nn.Module,
        *,
        example_inputs: Sequence[Any],
        memory: PlanMemory,
        partition: str,
        verbose: bool,
        artifacts: PlanningArtifacts,
        profiling_metadata: object,
    ) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.memory = memory
        self.partition = partition
        self.artifacts = artifacts
        self.profiling_metadata = profiling_metadata
        self.started = time.perf_counter_ns()
        self.timer = _PhaseTimer(verbose=verbose)

    def run(self) -> PlannedForward:
        """Execute the five explicit planning phases in dependency order."""

        request = self._validate_request()
        captured = self._capture_and_partition(request)
        profiled = self._compile_and_profile(request, captured)
        program = self._build_program(request, captured, profiled)
        selection = self._run_pressurefit(program)
        return self._admit_and_publish(
            request,
            captured,
            profiled,
            program,
            selection,
        )

    def _validate_request(self) -> _ForwardRequest:
        with self.timer.measure("validation"):
            _validate_forward_request(
                self.model,
                self.example_inputs,
                self.memory.execution_budget,
                self.memory.spill_budget,
            )
            signature = capture_input_signature(self.example_inputs)
            cpu_inputs = tuple(representative_cpu_inputs(self.example_inputs))
            _spill_pool_estimate(self.model, cpu_inputs, self.memory.spill_budget)
            workload = canonicalize_profiling_metadata(self.profiling_metadata)
        with self.timer.measure("runtime_binding"):
            installed = self.memory.installed
            device_ordinal = self.memory.execution_device
        return _ForwardRequest(
            signature,
            cpu_inputs,
            workload,
            installed,
            device_ordinal,
        )

    def _capture_and_partition(
        self,
        request: _ForwardRequest,
    ) -> _ForwardCaptureArtifacts:
        with self.timer.measure("capture_lowering"):
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_model = fake_cuda_model(
                self.model,
                fake_mode,
                device_index=request.device_ordinal,
            )
            fake_inputs = fake_cuda_inputs(
                request.cpu_inputs,
                fake_mode,
                device_index=request.device_ordinal,
            )
            with fake_mode, torch.no_grad():
                example_output = fake_model(*fake_inputs)
                _output_leaves, output_tree_spec = tree_flatten(example_output)
                capture = capture_forward(fake_model, fake_inputs)
            with self.timer.measure("export_archival"):
                self.artifacts.archive_export(
                    capture,
                    mode="forward",
                    position=0,
                )
            representative_roots = tuple(
                value.detach() if isinstance(value, torch.Tensor) else value
                for value in flat_runtime_arguments(
                    capture,
                    self.model,
                    request.cpu_inputs,
                )
            )
            with fake_mode, torch.no_grad():
                partitioned = partition_export(
                    capture,
                    fake_model,
                    partition=self.partition,
                    representative_root_inputs=representative_roots,
                )
                task_artifacts = capture_forward_stages(partitioned)
        return _ForwardCaptureArtifacts(
            fake_model,
            capture,
            partitioned,
            task_artifacts,
            output_tree_spec,
        )

    def _compile_and_profile(
        self,
        request: _ForwardRequest,
        captured: _ForwardCaptureArtifacts,
    ) -> _ForwardProfileArtifacts:
        profiler = CudaTaskProfiler(
            request.installed.library,
            device_ordinal=request.device_ordinal,
        )
        environment = profile_environment(
            device_ordinal=request.device_ordinal,
            provider_id="shadowspill.device_pool",
            implementation_revision=self.artifacts.cache.implementation_revision,
        )
        with self.timer.measure("compiler_manifest"):
            manifests = resolve_task_manifests(
                captured.task_artifacts,
                environment=environment,
                profile_cache=self.artifacts.profiles,
                profiler=profiler,
                progress=lambda index, total, state, digest: self.timer.progress(
                    f"compiled manifest {index}/{total} {state}: {digest[:12]}"
                ),
            )
        with self.timer.measure("structural_profiling"):
            profiles = profile_unique_artifacts(
                captured.task_artifacts,
                environment=environment,
                measure=profiler.measure,
                cache=self.artifacts.profiles,
                validate=lambda artifact, measurement: validate_compiled_profile(
                    artifact,
                    measurement,
                    manifests.manifests,
                ),
                progress=lambda index, total, state, digest: self.timer.progress(
                    f"structural profile {index}/{total} {state}: {digest[:12]}"
                ),
                profiling_metadata_digests=(request.workload.digest,)
                * len(captured.task_artifacts),
            )
        with self.timer.measure("compilation"):
            compiled_tasks = profiler.take_compiled_tasks(
                captured.task_artifacts,
                progress=lambda index, total, state, digest: self.timer.progress(
                    f"compiled entrypoint {index}/{total} {state}: {digest[:12]}"
                ),
            )
            _verify_manifest_identity(manifests, compiled_tasks)
            request.installed.library.shadowspill_pytorch_allocator_wait_idle()
        self.timer.attribute_compilation_and_profiling(profiler)
        return _ForwardProfileArtifacts(
            profiler,
            manifests,
            profiles,
            compiled_tasks,
        )

    def _build_program(
        self,
        request: _ForwardRequest,
        captured: _ForwardCaptureArtifacts,
        profiled: _ForwardProfileArtifacts,
    ) -> _ForwardProgramArtifacts:
        with self.timer.measure("program_lowering"):
            measurements = {
                artifact.compatibility_digest: measurement
                for artifact, measurement in zip(
                    captured.task_artifacts,
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
            lowered = lower_forward_program(
                captured.fake_model,
                captured.partitioned,
                captured.task_artifacts,
                profiled.profiles.measurements,
                storage_contracts={
                    digest: manifest.storage_contract
                    for digest, manifest in profiled.manifests.manifests.items()
                },
                compiled_root_allocations={
                    digest: manifest.root_allocations
                    for digest, manifest in profiled.manifests.manifests.items()
                },
                device_ordinal=request.device_ordinal,
                profile_compatibility_digests=profiled.profiles.key_digests,
            )
            workspace_reserve = _workspace_reserve(profiled.profiles.measurements)
            simulation_config = _simulation_config(
                self.memory,
                workspace_reserve,
                profiled.profiles,
            )
        return _ForwardProgramArtifacts(
            lowered,
            measurements,
            measurements_by_profile,
            workspace_reserve,
            simulation_config,
        )

    def _run_pressurefit(
        self,
        program: _ForwardProgramArtifacts,
    ) -> CachedPressureFitResult:
        with self.timer.measure("pressurefit_simulation"):
            return self.artifacts.select(
                program.lowered.program,
                initial_residency=program.lowered.initial_residency,
                final_residency=program.lowered.final_residency,
                config=program.simulation_config,
            )

    def _admit_and_publish(
        self,
        request: _ForwardRequest,
        captured: _ForwardCaptureArtifacts,
        profiled: _ForwardProfileArtifacts,
        program: _ForwardProgramArtifacts,
        selection: CachedPressureFitResult,
    ) -> PlannedForward:
        selected = selection.result
        with self.timer.measure("host_admission"):
            _reconcile_spill_pool(
                request.installed,
                predicted_host_peak=selected.simulation.host_peak_bytes,
                host_budget=self.memory.spill_budget,
            )
        slab_replay = self._replay_slab(request, profiled, program, selection)
        admission = _physical_admission(
            self.memory,
            request.installed,
            workspace_reserve=program.workspace_reserve,
            slab_replay=slab_replay,
        )
        entrypoints = tuple(
            EntrypointSpec(
                task_id=item.task_id,
                entrypoint_id=f"entrypoint_{index:06d}",
                executor_id="pytorch_inductor",
                abi_digest=item.artifact.compatibility_digest,
            )
            for index, item in enumerate(program.lowered.entrypoints)
        )
        execution_plan = selected.to_execution_plan(
            entrypoints=entrypoints,
            admission=admission,
        )
        bridge = RuntimeBridge(request.installed.library, execution_plan.program)
        state: MaterializedForwardState | None = None
        try:
            with self.timer.measure("materialization"):
                state = MaterializedForwardState(
                    self.model,
                    program.lowered,
                    captured.capture,
                    request.cpu_inputs,
                    bridge,
                    device_ordinal=request.device_ordinal,
                )
            with self.timer.measure("physical_sealing"):
                _seal_physical_budget(request.installed, execution_plan)
            with self.timer.measure("diagnostic_inventory"):
                task_stage_map, unique_stages = forward_stage_inventory(
                    program.lowered,
                    execution_plan,
                    program.measurements,
                    profiled.compiled_tasks.manifests,
                    profiling_metadata_digest=request.workload.digest,
                )
            with self.timer.measure("callable_construction"):
                executor = ForwardExecutor(
                    captured.partitioned,
                    program.lowered,
                    execution_plan,
                    bridge,
                    state,
                    profiled.compiled_tasks.functions,
                    captured.capture.user_output_indices,
                    captured.output_tree_spec,
                )
            report = _forward_report(
                request.signature.digest,
                execution_plan,
                profiled.profiles,
                tuple(self.timer.values),
                self.started,
                recomputation_cache_hit=selection.cache_hit,
                pressurefit_results=(selected,),
                captured_stage_count=len(captured.partitioned.stages),
                aot_unique_stage_abis=profiled.profiles.unique_keys,
                task_stage_map=task_stage_map,
                unique_stages=unique_stages,
                compiler_phase_timings_ns=(
                    profiled.profiler.compilation_phase_timings_ns
                ),
                compiler_phase_timings_by_abi=(
                    profiled.profiler.compilation_phase_timings_by_abi
                ),
                cache_directories=self.artifacts.cache.diagnostics(),
                cache_artifacts=_public_cache_artifacts(self.artifacts.cache),
                profiling_metadata=(request.workload,),
                memory=self.memory,
            )
            report = _finalize_plan_report(
                self.model,
                report,
                self.artifacts.cache,
                started=self.started,
            )
            return PlannedForward(
                self.model,
                request.signature,
                executor,
                state,
                report,
                self.memory.runtime,
            )
        except BaseException:
            if state is not None:
                state.restore_cpu_and_unregister()
            raise

    def _replay_slab(
        self,
        request: _ForwardRequest,
        profiled: _ForwardProfileArtifacts,
        program: _ForwardProgramArtifacts,
        selection: CachedPressureFitResult,
    ) -> SlabReplay:
        with self.timer.measure("slab_admission"):
            try:
                return replay_selected_schedule(
                    selection.result,
                    program.measurements_by_profile,
                    execution_pool_bytes=(
                        int(request.installed.admission.execution_pool_bytes)
                        - profiled.profiles.fixed_slab_bytes
                    ),
                    output_bindings=output_bindings_for_entrypoints(
                        selection.result.program.selected_tasks(
                            selection.result.selections
                        ),
                        program.lowered.entrypoints,
                        {
                            item.object_id: item.alias_group_id
                            for item in selection.result.program.objects
                        },
                    ),
                )
            except AdmissionError as exc:
                raise PlanningError(f"slab spatial admission failed: {exc}") from exc


def _verify_manifest_identity(
    resolved: ResolvedTaskManifests,
    compiled: CompiledTaskSet,
) -> None:
    for digest, manifest in compiled.manifests.items():
        expected = resolved.manifests.get(digest)
        if expected is None or (
            expected.compatibility_digest != manifest.compatibility_digest
        ):
            raise PlanningError(
                f"compiled entrypoint changed its storage ABI: artifact={digest}"
            )


def _validate_forward_request(
    model: nn.Module,
    example_inputs: Sequence[Any],
    execution_budget: int,
    spill_budget: int,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(example_inputs, (list, tuple)):
        raise PlanningError("example_inputs must be a list or tuple")
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
            raise PlanningError(
                f"registered tensor {name!r} must be CPU resident before planning"
            )


def _spill_pool_estimate(
    model: nn.Module, example_inputs: object, host_budget: int
) -> int:
    tensors = [
        tensor
        for _name, tensor in (
            *tuple(model.named_parameters(remove_duplicate=False)),
            *tuple(model.named_buffers(remove_duplicate=False)),
        )
    ]
    leaves, _ = tree_flatten(example_inputs)
    tensors.extend(value for value in leaves if isinstance(value, torch.Tensor))
    unique: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        unique[(tensor.device.type, storage._cdata)] = int(storage.nbytes())
    base = sum(unique.values())
    requested = _round_up(base + max(_HOST_LEEWAY_MINIMUM, base // 10), _HOST_ALIGNMENT)
    if requested > host_budget:
        raise PlanningError(
            "spill-pool budget cannot hold model/input storage plus admission leeway: "
            f"required={requested}, budget={host_budget}"
        )
    return requested


def _workspace_reserve(measurements: Sequence[Any]) -> int:
    peak = max((item.workspace_charged_bytes for item in measurements), default=0)
    return workspace_reserve_bytes(peak)


def _simulation_config(
    memory: PlanMemory,
    workspace_reserve: int,
    profiles: ProfilingResult,
) -> SimulationConfig:
    """Build the exact framework-neutral simulator input for one Program."""

    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=_simulation_capacity(
            memory.execution_budget,
            workspace_reserve,
            profiles.measurements,
            fixed_slab_bytes=profiles.fixed_slab_bytes,
        ),
        host_capacity_bytes=memory.spill_budget,
        fetch_bandwidth_bytes_per_second=memory.transfers.route(
            memory.spill.name,
            memory.execution.name,
        ).bandwidth_bytes_per_second,
        evict_bandwidth_bytes_per_second=memory.transfers.route(
            memory.execution.name,
            memory.spill.name,
        ).bandwidth_bytes_per_second,
        fetch_latency_ns=memory.transfers.route(
            memory.spill.name,
            memory.execution.name,
        ).latency_nanoseconds,
        evict_latency_ns=memory.transfers.route(
            memory.execution.name,
            memory.spill.name,
        ).latency_nanoseconds,
    )


def _physical_admission(
    memory: PlanMemory,
    installed: InstalledAllocator,
    *,
    workspace_reserve: int,
    slab_replay: SlabReplay,
) -> PhysicalAdmission:
    """Describe the physical resources admitted after spatial replay."""

    return PhysicalAdmission(
        device_budget_bytes=(
            memory.execution.physical_capacity or memory.execution_budget
        ),
        host_budget_bytes=memory.spill_budget,
        context_bytes=int(installed.admission.context_bytes),
        provider_headroom_bytes=int(installed.admission.provider_headroom_bytes),
        slab_bytes=memory.execution_budget,
        workspace_reserve_bytes=workspace_reserve,
        host_reservation_bytes=int(installed.admission.spill_pool_bytes),
        predicted_fragmentation_bytes=slab_replay.peak_fragmentation_bytes,
    )


def _reconcile_spill_pool(
    installed: InstalledAllocator,
    *,
    predicted_host_peak: int,
    host_budget: int,
) -> None:
    del installed
    if predicted_host_peak < 0:
        raise PlanningError("predicted host peak must be non-negative")
    if predicted_host_peak > host_budget:
        raise PlanningError(
            "predicted spill-pool peak exceeds the selected plan budget: "
            f"peak={predicted_host_peak}, budget={host_budget}"
        )


def _simulation_capacity(
    execution_pool_bytes: int,
    workspace_reserve: int,
    measurements: Sequence[Any],
    *,
    fixed_slab_bytes: int = 0,
) -> int:
    usable_slab = execution_pool_bytes - fixed_slab_bytes
    if fixed_slab_bytes < 0 or usable_slab < 0:
        raise PlanningError(
            "fixed provider allocations exceed the admitted slab: "
            f"slab={execution_pool_bytes}, fixed={fixed_slab_bytes}"
        )
    if workspace_reserve > usable_slab:
        raise PlanningError(
            "the admitted slab is smaller than the workspace reserve: "
            f"usable_slab={usable_slab}, reserve={workspace_reserve}"
        )
    maximum_workspace = max(
        (item.workspace_charged_bytes for item in measurements), default=0
    )
    return usable_slab - workspace_reserve + maximum_workspace


def _seal_physical_budget(installed: InstalledAllocator, execution_plan: Any) -> None:
    library = installed.library
    status = int(library.shadowspill_pytorch_check_physical_budget())
    if status != 0:
        raise PlanningError(
            f"provider allocations exceeded physical admission (status {status})"
        )
    statistics = AdapterStatistics()
    status = int(
        library.shadowspill_pytorch_allocator_statistics(ctypes.byref(statistics))
    )
    if status != 0:
        raise PlanningError(f"allocator statistics failed with status {status}")
    required = max(
        int(installed.admission.provider_headroom_bytes),
        _round_up(
            int(statistics.observed_external_high_water_bytes) + 64 * _MIB,
            64 * _MIB,
        ),
    )
    # Every selected task can contribute one shared completion fence. Every
    # admitted transfer may also remain queued on its directed transfer lane
    # with a distinct readiness/completion event. The streams preserve FIFO
    # order; keeping the complete admitted window enqueued lets before_task
    # insert stream waits without blocking the Python dispatch thread.
    initial_transfer_count = sum(
        item.location.value == "device"
        for item in execution_plan.schedule.initial_residency
    )
    scheduled_transfer_count = sum(
        item.kind in {MemoryActionKind.OFFLOAD, MemoryActionKind.PREFETCH}
        for item in execution_plan.schedule.actions
    )
    event_pool_reserve = max(
        256,
        len(execution_plan.program.selected_tasks(execution_plan.selections))
        + initial_transfer_count
        + scheduled_transfer_count
        + 2 * int(statistics.cuda.event_pool_peak_in_use)
        + 64,
    )
    status = int(
        library.shadowspill_pytorch_seal_physical_budget(required, event_pool_reserve)
    )
    if status != 0:
        reserved = int(installed.admission.provider_headroom_bytes)
        raise PlanningError(
            "observed provider memory exceeds the reserved headroom: "
            f"required={required}, reserved={reserved}"
        )


def _forward_report(
    signature_digest: str,
    execution_plan: Any,
    profiles: Any,
    timings: tuple[tuple[str, int], ...],
    started: int,
    *,
    recomputation_cache_hit: bool = False,
    pressurefit_results: tuple[Any, ...] = (),
    captured_stage_count: int = 0,
    aot_unique_stage_abis: int = 0,
    aot_graph_pair_cache_hits: int = 0,
    aot_graph_pair_cache_misses: int = 0,
    task_stage_map: tuple[PlanTaskStage, ...] = (),
    unique_stages: tuple[PlanUniqueStage, ...] = (),
    compiler_phase_timings_ns: tuple[tuple[str, int], ...] = (),
    compiler_phase_timings_by_abi: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ] = (),
    cache_directories: tuple[tuple[str, str], ...] = (),
    cache_artifacts: tuple[PlanCacheArtifact, ...] = (),
    profiling_metadata: tuple[Any, ...] = (),
    memory: PlanMemory,
) -> PlanReport:
    identity = {
        "mode": "forward",
        "signature": signature_digest,
        "artifacts": [
            entrypoint.abi_digest for entrypoint in execution_plan.entrypoints
        ],
        "profiling_metadata": [item.digest for item in profiling_metadata],
    }
    capture_identity = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sizes = {
        item.alias_group_id: item.size_bytes
        for item in execution_plan.program.alias_groups
    }
    actions = execution_plan.schedule.actions
    elapsed = time.perf_counter_ns() - started
    # Training historically retained the aggregate ``capture_lowering`` timing
    # as well as its three children for report compatibility. The structured
    # diagnostic publishes only mutually exclusive leaves.
    nested_capture = any(
        name
        in {
            "objective_export",
            "export_archival",
            "stage_partition_aot",
            "storage_layout_lowering",
        }
        for name, _ in timings
    )
    phases = tuple(
        PlanPhaseTiming(name, duration)
        for name, duration in timings
        if not (nested_capture and name == "capture_lowering")
    )
    measured = sum(item.duration_ns for item in phases)
    if measured > elapsed:
        raise RuntimeError(
            "plan phase intervals overlap: measured phase time exceeds wall time"
        )
    diagnostics = PlanDiagnostics(
        phases=phases,
        total_wall_time_ns=elapsed,
        unattributed_overhead_ns=elapsed - measured,
        profile_unique_keys=profiles.unique_keys,
        profile_cache_hits=profiles.cache_hits,
        profile_cache_misses=profiles.cache_misses,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        recomputation_cache_hits=int(recomputation_cache_hit),
        recomputation_cache_misses=int(not recomputation_cache_hit),
        task_stage_map=task_stage_map,
        unique_stages=unique_stages,
        compiler_phase_timings_ns=compiler_phase_timings_ns,
        compiler_profiles=tuple(
            PlanCompilerProfile(
                structural_abi_key,
                tuple(PlanPhaseTiming(name, duration) for name, duration in values),
            )
            for structural_abi_key, values in compiler_phase_timings_by_abi
        ),
        cache_directories=cache_directories,
        cache_artifacts=cache_artifacts,
        profiling_metadata=tuple(
            PlanProfilingMetadata(index, item.digest, item.canonical_json)
            for index, item in enumerate(profiling_metadata)
        ),
    )
    return PlanReport(
        mode="forward",
        capture_identity=capture_identity,
        execution_plan=execution_plan,
        task_profiles=execution_plan.program.profiles,
        transfer_actions=actions,
        transfer_bytes_evicted=sum(
            sizes[item.alias_group_id]
            for item in actions
            if item.kind is MemoryActionKind.OFFLOAD
        ),
        transfer_bytes_fetched=sum(
            sizes[item.alias_group_id]
            for item in actions
            if item.kind is MemoryActionKind.PREFETCH
        ),
        profile_unique_keys=profiles.unique_keys,
        profile_cache_hits=profiles.cache_hits,
        profile_cache_misses=profiles.cache_misses,
        profiling_provenance=tuple(
            dict.fromkeys(item.provenance for item in profiles.measurements)
        ),
        phase_timings_ns=(*timings, ("total", elapsed)),
        recomputation_cache_hits=int(recomputation_cache_hit),
        recomputation_cache_misses=int(not recomputation_cache_hit),
        fixed_slab_bytes=profiles.fixed_slab_bytes,
        pressurefit_results=pressurefit_results,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        diagnostics=diagnostics,
        execution_pool=memory.execution.name,
        spill_pool=memory.spill.name,
        execution_budget_bytes=memory.execution_budget,
        spill_budget_bytes=memory.spill_budget,
        execution_device=memory.execution_device,
        transfer_capabilities=memory.transfers,
    )


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _public_cache_artifacts(
    planning_cache: PlanningCache,
) -> tuple[PlanCacheArtifact, ...]:
    return tuple(
        PlanCacheArtifact(
            item.category,
            item.kind,
            item.digest,
            str(item.path),
            item.access,
            item.schema,
            item.dependencies,
        )
        for item in planning_cache.artifacts()
    )


def _finalize_plan_report(
    model: nn.Module,
    report: PlanReport,
    planning_cache: PlanningCache,
    *,
    started: int,
) -> PlanReport:
    archival_started = time.perf_counter_ns()
    artifacts_before_plan = _public_cache_artifacts(planning_cache)
    planning_cache.archive_plan(
        model_label=f"{type(model).__module__}.{type(model).__qualname__}",
        capture_identity=report.capture_identity,
        execution_plan=report.execution_plan,
        initial_execution_plan=report.initial_execution_plan,
        manifest={
            "mode": report.mode,
            "execution_pool": report.execution_pool,
            "spill_pool": report.spill_pool,
            "execution_budget_bytes": report.execution_budget_bytes,
            "spill_budget_bytes": report.spill_budget_bytes,
            "execution_device": report.execution_device,
            "implementation_revision": planning_cache.implementation_revision,
            "phase_timings_ns": [list(item) for item in report.phase_timings_ns],
            "artifacts": [item.as_dict() for item in artifacts_before_plan],
        },
    )
    archival_duration = time.perf_counter_ns() - archival_started
    elapsed = time.perf_counter_ns() - started
    phases = (
        *report.diagnostics.phases,
        PlanPhaseTiming("plan_archival", archival_duration),
    )
    measured = sum(item.duration_ns for item in phases)
    if measured > elapsed:
        raise RuntimeError(
            "plan phase intervals overlap after archival: measured time exceeds wall"
        )
    diagnostics = replace(
        report.diagnostics,
        phases=phases,
        total_wall_time_ns=elapsed,
        unattributed_overhead_ns=elapsed - measured,
        cache_artifacts=_public_cache_artifacts(planning_cache),
    )
    phase_timings = tuple(
        item for item in report.phase_timings_ns if item[0] != "total"
    )
    return replace(
        report,
        diagnostics=diagnostics,
        phase_timings_ns=(
            *phase_timings,
            ("plan_archival", archival_duration),
            ("total", elapsed),
        ),
    )


__all__ = ["build_forward"]
