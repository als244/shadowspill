"""Small shared helpers for composable PyTorch planning phases."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten

from shadowspill.planner import (
    PressureFitInfeasibleError,
    PressureFitSearchExhaustedError,
)
from shadowspill.pytorch.profiling import (
    ProfilingResult,
    TaskMeasurement,
)
from shadowspill.pytorch.profiling.profiler import CudaTaskProfiler
from shadowspill.runtime import workspace_reserve_bytes
from shadowspill.simulator import SimulationConfig

from ..contracts import (
    AdmissionError,
    PlanInfeasibleError,
    PlanningError,
    PlanSearchExhaustedError,
)
from ..runtime_adapter import PlanMemory

_MIB = 1 << 20
_SPILL_LEEWAY_MINIMUM = 256 * _MIB
_SPILL_ALIGNMENT = 64 << 10


class PlanningTimer:
    """Collect non-overlapping phase timings and optional progress output."""

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

    def attribute_compilation_and_profiling(
        self,
        profiler: CudaTaskProfiler,
    ) -> None:
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
        compilation = (
            profiler.compilation_wall_time_ns
            - profiler.saved_control_compilation_wall_time_ns
        )
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
    def __init__(self, timer: PlanningTimer, name: str) -> None:
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
        error = exception[1]
        if isinstance(error, BaseException):
            error.add_note(
                "ShadowSpill planning phase "
                f"{self._name!r} failed after {duration / 1e9:.3f} seconds"
            )


def validate_cpu_model(model: nn.Module) -> None:
    """Require initialized CPU registrations before allocator materialization."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    for name, tensor in tuple(model.named_parameters()) + tuple(model.named_buffers()):
        if tensor.device.type != "cpu":
            raise PlanningError(
                f"registered tensor {name!r} must be CPU resident before planning"
            )


def validate_budgets(execution_budget: int, spill_budget: int) -> None:
    """Validate public physical byte limits."""

    if isinstance(execution_budget, bool) or not isinstance(execution_budget, int):
        raise TypeError("execution_budget must be an integer byte count")
    if isinstance(spill_budget, bool) or not isinstance(spill_budget, int):
        raise TypeError("spill_budget must be an integer byte count")
    if execution_budget <= 0:
        raise AdmissionError("execution_budget must be positive")
    if spill_budget <= 0:
        raise AdmissionError("spill_budget must be positive")


def estimate_spill_reservation(
    model: nn.Module,
    example_inputs: object,
    spill_budget: int,
) -> int:
    """Conservatively validate that spill storage can hold state and inputs."""

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
    requested = round_up(
        base + max(_SPILL_LEEWAY_MINIMUM, base // 10),
        _SPILL_ALIGNMENT,
    )
    if requested > spill_budget:
        raise AdmissionError(
            "spill-pool budget cannot hold model/input storage plus admission "
            f"leeway: required={requested}, budget={spill_budget}"
        )
    return requested


def workspace_reserve(measurements: Sequence[TaskMeasurement]) -> int:
    """Return the conservative global workspace reserve."""

    peak = max((item.workspace_charged_bytes for item in measurements), default=0)
    return workspace_reserve_bytes(peak)


def simulation_capacity(
    execution_pool_bytes: int,
    workspace_reserve_bytes_: int,
    measurements: Sequence[TaskMeasurement],
    *,
    fixed_slab_bytes: int = 0,
) -> int:
    """Translate a physical slab admission into PressureFit object capacity."""

    usable_slab = execution_pool_bytes - fixed_slab_bytes
    if fixed_slab_bytes < 0 or usable_slab < 0:
        raise AdmissionError(
            "fixed provider allocations exceed the admitted slab: "
            f"slab={execution_pool_bytes}, fixed={fixed_slab_bytes}"
        )
    if workspace_reserve_bytes_ > usable_slab:
        raise AdmissionError(
            "the admitted slab is smaller than the workspace reserve: "
            f"usable_slab={usable_slab}, reserve={workspace_reserve_bytes_}"
        )
    maximum_workspace = max(
        (item.workspace_charged_bytes for item in measurements), default=0
    )
    capacity = usable_slab - workspace_reserve_bytes_ + maximum_workspace
    if capacity <= 0:
        raise public_infeasible_plan_error(
            PressureFitInfeasibleError(
                "the admitted slab leaves no capacity for Program objects",
                kind="analytic_capacity",
                required_bytes=1,
                capacity_bytes=max(0, capacity),
            )
        )
    return capacity


def fixed_execution_bytes(memory: PlanMemory, profiles: ProfilingResult) -> int:
    """Return every process-persistent byte carved from the execution slab."""

    return memory.installed.fixed_execution_bytes + profiles.fixed_slab_bytes


def build_simulation_config(
    memory: PlanMemory,
    workspace_reserve_bytes_: int,
    profiles: ProfilingResult,
) -> SimulationConfig:
    """Build the exact framework-neutral simulator input for one Program."""

    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=simulation_capacity(
            memory.execution_budget,
            workspace_reserve_bytes_,
            profiles.measurements,
            fixed_slab_bytes=fixed_execution_bytes(memory, profiles),
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


def public_infeasible_plan_error(
    error: PressureFitInfeasibleError,
) -> PlanInfeasibleError:
    """Preserve PressureFit's structured infeasibility at the public boundary."""

    fields = [
        "ShadowSpill could not construct a feasible memory schedule",
        f"constraint: {error.kind}",
    ]
    if error.device_id is not None:
        fields.append(f"device: {error.device_id}")
    if error.boundary_task_id is not None:
        fields.append(f"boundary_task: {error.boundary_task_id}")
    if error.required_bytes is not None:
        fields.append(f"required: {error.required_bytes}")
    if error.capacity_bytes is not None:
        fields.append(f"capacity: {error.capacity_bytes}")
    fields.append(f"detail: {error}")
    return PlanInfeasibleError(
        "\n".join(fields),
        kind=error.kind,
        device_id=error.device_id,
        boundary_task_id=error.boundary_task_id,
        required_bytes=error.required_bytes,
        capacity_bytes=error.capacity_bytes,
    )


def public_search_exhausted_error(
    error: PressureFitSearchExhaustedError,
) -> PlanSearchExhaustedError:
    """Distinguish an evaluation ceiling from proof of plan infeasibility."""

    exhausted = tuple(
        item for item in error.diagnostics if item.status == "exhausted"
    )
    largest_repairs = max(
        (item.repairs.total_attempts for item in exhausted),
        default=0,
    )
    return PlanSearchExhaustedError(
        "\n".join(
            (
                "ShadowSpill planning stopped at its bounded repair ceiling",
                f"exhausted_candidates: {len(exhausted)}",
                f"largest_repair_count: {largest_repairs}",
                "detail: no exhausted candidate was classified as physically "
                "infeasible",
            )
        )
    )


def round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


__all__ = [
    "PlanningTimer",
    "build_simulation_config",
    "estimate_spill_reservation",
    "public_infeasible_plan_error",
    "public_search_exhausted_error",
    "simulation_capacity",
    "validate_budgets",
    "validate_cpu_model",
    "workspace_reserve",
]
