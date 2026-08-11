"""Rollback-safe construction shared by ShadowSpill PyTorch entrypoints."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten

from shadowspill.ir import EntrypointSpec, MemoryActionKind, PhysicalAdmission
from shadowspill.planner._cache import PressureFitCache
from shadowspill.runtime import AdmissionError, workspace_reserve_bytes
from shadowspill.simulator import SimulationConfig

from ._abi import AdapterStatistics
from ._allocator import (
    AllocatorInstallError,
    InstalledAllocator,
    install_allocator,
    installed_allocator,
    resize_host_arena,
)
from .aot import capture_forward
from .compiler import CudaTaskProfiler, profile_environment
from .contracts import PlanningError
from .executor import ForwardExecutor
from .fake import fake_cuda_inputs, fake_cuda_model
from .guards import capture_input_signature
from .lowering import lower_forward_program
from .materialization import MaterializedForwardState, representative_cpu_inputs
from .partition import capture_forward_stages, partition_export
from .profiling import ProfileCache, profile_unique_artifacts
from .public import PlannedForward, PlanReport
from .runtime_bridge import RuntimeBridge
from .spatial_admission import (
    output_bindings_for_entrypoints,
    replay_selected_schedule,
)

_MIB = 1 << 20
_PROVIDER_HEADROOM = 512 * _MIB
_HOST_LEEWAY_MINIMUM = 256 * _MIB
_HOST_ALIGNMENT = 64 << 10


class _PhaseTimer:
    def __init__(self) -> None:
        self.values: list[tuple[str, int]] = []

    def measure(self, name: str) -> _MeasuredPhase:
        return _MeasuredPhase(self, name)


class _MeasuredPhase:
    def __init__(self, timer: _PhaseTimer, name: str) -> None:
        self._timer = timer
        self._name = name
        self._start = 0

    def __enter__(self) -> None:
        self._start = time.perf_counter_ns()

    def __exit__(self, *exception: object) -> None:
        del exception
        self._timer.values.append((self._name, time.perf_counter_ns() - self._start))


def build_forward(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    device_budget: int,
    host_budget: int,
    partition: str,
) -> PlannedForward:
    """Construct a planned forward callable without mutating arithmetic."""

    started = time.perf_counter_ns()
    timer = _PhaseTimer()
    with timer.measure("validation"):
        _validate_forward_request(model, example_inputs, device_budget, host_budget)
        signature = capture_input_signature(example_inputs)
        cpu_inputs = representative_cpu_inputs(example_inputs)
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
        fake_inputs = fake_cuda_inputs(cpu_inputs, fake_mode, device_index=0)
        with fake_mode, torch.no_grad():
            example_output = fake_model(*fake_inputs)
            _output_leaves, output_tree_spec = tree_flatten(example_output)
            capture = capture_forward(fake_model, fake_inputs)
            partitioned = partition_export(capture, fake_model, partition=partition)
            artifacts = capture_forward_stages(partitioned)

    profiler = CudaTaskProfiler(installed.library, device_ordinal=0)
    with timer.measure("structural_profiling"):
        environment = profile_environment(
            device_ordinal=0, provider_id="shadowspill.cuda_slab"
        )
        profiles = profile_unique_artifacts(
            artifacts,
            environment=environment,
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
        lowered = lower_forward_program(
            fake_model,
            partitioned,
            artifacts,
            profiles.measurements,
            device_ordinal=0,
        )
        workspace_reserve = _workspace_reserve(profiles.measurements)
        simulation_capacity = _simulation_capacity(
            int(installed.admission.slab_bytes),
            workspace_reserve,
            profiles.measurements,
            fixed_slab_bytes=profiles.fixed_slab_bytes,
        )
        simulation_config = SimulationConfig.single_device(
            "cuda_0",
            device_capacity_bytes=simulation_capacity,
            host_capacity_bytes=host_budget,
            h2d_bandwidth_bytes_per_second=_positive_environment_integer(
                "SHADOWSPILL_H2D_BANDWIDTH_BYTES_PER_SECOND", 24 << 30
            ),
            d2h_bandwidth_bytes_per_second=_positive_environment_integer(
                "SHADOWSPILL_D2H_BANDWIDTH_BYTES_PER_SECOND", 24 << 30
            ),
            h2d_latency_ns=_nonnegative_environment_integer(
                "SHADOWSPILL_H2D_LATENCY_NS", 5_000
            ),
            d2h_latency_ns=_nonnegative_environment_integer(
                "SHADOWSPILL_D2H_LATENCY_NS", 5_000
            ),
        )

    with timer.measure("pressurefit_simulation"):
        cached_selection = PressureFitCache().resolve(
            lowered.program,
            initial_residency=lowered.initial_residency,
            final_residency=lowered.final_residency,
            config=simulation_config,
        )
        selected = cached_selection.result

    with timer.measure("host_admission"):
        _reconcile_host_arena(
            installed,
            predicted_host_peak=selected.simulation.host_peak_bytes,
            host_budget=host_budget,
        )

    with timer.measure("slab_admission"):
        try:
            slab_replay = replay_selected_schedule(
                selected,
                measurements,
                slab_bytes=(
                    int(installed.admission.slab_bytes) - profiles.fixed_slab_bytes
                ),
                output_bindings=output_bindings_for_entrypoints(
                    selected.program.selected_tasks(selected.selections),
                    lowered.entrypoints,
                    {
                        item.object_id: item.alias_group_id
                        for item in selected.program.objects
                    },
                ),
            )
        except AdmissionError as exc:
            raise PlanningError(f"slab spatial admission failed: {exc}") from exc

    admission = PhysicalAdmission(
        device_budget_bytes=device_budget,
        host_budget_bytes=host_budget,
        context_bytes=int(installed.admission.context_bytes),
        provider_headroom_bytes=int(installed.admission.provider_headroom_bytes),
        slab_bytes=int(installed.admission.slab_bytes),
        workspace_reserve_bytes=workspace_reserve,
        host_reservation_bytes=int(installed.admission.host_arena_bytes),
        predicted_fragmentation_bytes=slab_replay.peak_fragmentation_bytes,
    )
    entrypoints = tuple(
        EntrypointSpec(
            task_id=item.task_id,
            entrypoint_id=f"entrypoint_{index:06d}",
            executor_id="pytorch_inductor",
            abi_digest=item.artifact.compatibility_digest,
        )
        for index, item in enumerate(lowered.entrypoints)
    )
    execution_plan = selected.to_execution_plan(
        entrypoints=entrypoints, admission=admission
    )

    bridge = RuntimeBridge(installed.library, execution_plan.program)
    state: MaterializedForwardState | None = None
    try:
        with timer.measure("materialization"):
            state = MaterializedForwardState(
                model,
                lowered,
                capture,
                cpu_inputs,
                bridge,
                device_ordinal=0,
            )
        with timer.measure("physical_sealing"):
            _seal_physical_budget(installed)
        with timer.measure("callable_construction"):
            executor = ForwardExecutor(
                partitioned,
                lowered,
                execution_plan,
                bridge,
                state,
                functions,
                capture.user_output_indices,
                output_tree_spec,
            )
            report = _forward_report(
                signature.digest,
                execution_plan,
                profiles,
                tuple(timer.values),
                started,
                recomputation_cache_hit=cached_selection.cache_hit,
            )
            return PlannedForward(model, signature, executor, state, report)
    except BaseException:
        if state is not None:
            state.restore_cpu_and_unregister()
        raise


def _validate_forward_request(
    model: nn.Module,
    example_inputs: Sequence[Any],
    device_budget: int,
    host_budget: int,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(example_inputs, (list, tuple)):
        raise PlanningError("example_inputs must be a list or tuple")
    if isinstance(device_budget, bool) or not isinstance(device_budget, int):
        raise TypeError("device_budget must be an integer byte count")
    if isinstance(host_budget, bool) or not isinstance(host_budget, int):
        raise TypeError("host_budget must be an integer byte count")
    if device_budget <= _PROVIDER_HEADROOM:
        raise PlanningError("device_budget must exceed the provider headroom")
    if host_budget <= 0:
        raise PlanningError("host_budget must be positive")
    for name, tensor in tuple(model.named_parameters()) + tuple(model.named_buffers()):
        if tensor.device.type != "cpu":
            raise PlanningError(
                f"registered tensor {name!r} must be CPU resident before planning"
            )


def _host_arena_estimate(
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
            "host budget cannot hold model/input backing plus admission leeway: "
            f"required={requested}, budget={host_budget}"
        )
    return requested


def _ensure_allocator(
    *, device_budget: int, host_arena: int, device_ordinal: int
) -> InstalledAllocator:
    current = installed_allocator()
    path = _adapter_path()
    if current is None:
        return install_allocator(
            path,
            device_ordinal=device_ordinal,
            device_budget_bytes=device_budget,
            provider_headroom_bytes=_PROVIDER_HEADROOM,
            host_arena_bytes=host_arena,
        )
    if (
        current.path != path
        or int(current.admission.device_ordinal) != device_ordinal
        or int(current.admission.device_budget_bytes) != device_budget
        or int(current.admission.host_arena_bytes) < host_arena
    ):
        raise PlanningError(
            "the process-global ShadowSpill allocator has incompatible admission"
        )
    return current


def _adapter_path() -> Path:
    configured = os.environ.get("SHADOWSPILL_PYTORCH_LIBRARY")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parents[1] / "lib" / "libshadowspill_pytorch.so"
    if not path.is_file():
        raise PlanningError(
            "ShadowSpill's PyTorch adapter was not found; set "
            f"SHADOWSPILL_PYTORCH_LIBRARY (looked for {path})"
        )
    return path


def _workspace_reserve(measurements: Sequence[Any]) -> int:
    peak = max((item.workspace_charged_bytes for item in measurements), default=0)
    return workspace_reserve_bytes(peak)


def _reconcile_host_arena(
    installed: InstalledAllocator,
    *,
    predicted_host_peak: int,
    host_budget: int,
) -> None:
    if predicted_host_peak < 0:
        raise PlanningError("predicted host peak must be non-negative")
    requested = _round_up(
        predicted_host_peak + max(_HOST_LEEWAY_MINIMUM, predicted_host_peak // 10),
        _HOST_ALIGNMENT,
    )
    requested = max(requested, int(installed.admission.host_arena_bytes))
    try:
        resize_host_arena(
            installed,
            host_arena_bytes=requested,
            host_budget_bytes=host_budget,
        )
    except AllocatorInstallError as exc:
        raise PlanningError(str(exc)) from exc


def _simulation_capacity(
    slab_bytes: int,
    workspace_reserve: int,
    measurements: Sequence[Any],
    *,
    fixed_slab_bytes: int = 0,
) -> int:
    usable_slab = slab_bytes - fixed_slab_bytes
    if fixed_slab_bytes < 0 or usable_slab < 0:
        raise PlanningError(
            "fixed provider allocations exceed the admitted slab: "
            f"slab={slab_bytes}, fixed={fixed_slab_bytes}"
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


def _seal_physical_budget(installed: InstalledAllocator) -> None:
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
        _PROVIDER_HEADROOM,
        _round_up(
            int(statistics.observed_external_high_water_bytes) + 64 * _MIB,
            64 * _MIB,
        ),
    )
    status = int(library.shadowspill_pytorch_seal_physical_budget(required))
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
) -> PlanReport:
    identity = {
        "mode": "forward",
        "signature": signature_digest,
        "artifacts": [
            entrypoint.abi_digest for entrypoint in execution_plan.entrypoints
        ],
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
    return PlanReport(
        mode="forward",
        capture_identity=capture_identity,
        execution_plan=execution_plan,
        task_profiles=execution_plan.program.profiles,
        transfer_actions=actions,
        transfer_bytes_to_host=sum(
            sizes[item.alias_group_id]
            for item in actions
            if item.kind is MemoryActionKind.OFFLOAD
        ),
        transfer_bytes_to_device=sum(
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
    )


def _positive_environment_integer(name: str, default: int) -> int:
    value = _nonnegative_environment_integer(name, default)
    if value == 0:
        raise PlanningError(f"{name} must be positive")
    return value


def _nonnegative_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise PlanningError(f"{name} must be an integer") from exc
    if value < 0:
        raise PlanningError(f"{name} must be non-negative")
    return value


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


__all__ = ["build_forward"]
