"""Compilation and isolated CUDA measurement for one structural task ABI."""

from __future__ import annotations

import copy
import ctypes
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch._inductor.compile_fx import compile_fx
from torch.utils._pytree import tree_flatten

from ._abi import AdapterStatistics, Allocation
from ._telemetry import (
    AllocationTelemetryError,
    read_allocation_telemetry,
    start_allocation_telemetry,
    stop_allocation_telemetry,
    summarize_task_workspace,
)
from .capture import GraphArtifact
from .contracts import CaptureError
from .optimizer import OpaqueOptimizerArtifact, materialize_opaque_optimizer
from .profiling import ProfilableArtifact, ProfileEnvironment, TaskMeasurement


@dataclass(frozen=True, slots=True)
class CompiledTask:
    """One compiled graph and allocator-owned representative arguments."""

    artifact: GraphArtifact
    function: Callable[..., object]
    example_arguments: tuple[object, ...]
    execution_provider: str = "torch-inductor"
    graph_node_count: int = 0

    def __call__(self) -> object:
        # GraphArtifact training derivatives are explicit AOT forward/backward
        # programs. Letting dispatcher autograd wrap a registered custom op
        # here would create a second, hidden saved-tensor lifetime.
        with torch.no_grad():
            return self.function(*self.example_arguments)


def profile_environment(*, device_ordinal: int, provider_id: str) -> ProfileEnvironment:
    """Describe every implementation attribute that can change task cost."""

    properties = torch.cuda.get_device_properties(device_ordinal)
    return ProfileEnvironment(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device_name=properties.name,
        compute_capability=(properties.major, properties.minor),
        compiler_id="shadowspill-structural-compiler/v2:torch-inductor",
        provider_id=provider_id,
    )


def materialize_example_arguments(
    arguments: Sequence[object], *, device_ordinal: int
) -> tuple[object, ...]:
    """Create real CUDA values while preserving tensor storage aliases."""

    cuda_device = torch.device("cuda", device_ordinal)
    storages: dict[tuple[int, str], torch.Tensor] = {}
    results: list[object] = []
    with torch.no_grad():
        for argument in arguments:
            if not isinstance(argument, torch.Tensor):
                results.append(copy.deepcopy(argument))
                continue
            if argument.layout is not torch.strided:
                raise CaptureError(
                    "compiled task examples currently require strided tensors"
                )
            source_storage = argument.untyped_storage()
            target_device = (
                cuda_device if argument.device.type == "cuda" else argument.device
            )
            key = (source_storage._cdata, str(target_device))
            storage = storages.get(key)
            if storage is None:
                storage = torch.empty(
                    source_storage.nbytes(), dtype=torch.uint8, device=target_device
                )
                storage.zero_()
                storages[key] = storage
            tensor = torch.empty(0, dtype=argument.dtype, device=target_device)
            tensor.set_(
                storage.untyped_storage(),
                argument.storage_offset(),
                tuple(argument.shape),
                tuple(argument.stride()),
            )
            tensor.requires_grad_(argument.requires_grad)
            results.append(tensor)
    return tuple(results)


def compile_artifact(
    artifact: GraphArtifact,
    *,
    device_ordinal: int,
    representative_arguments: Sequence[object] | None = None,
) -> CompiledTask:
    """Compile one explicit FX task against real allocator-owned CUDA values."""

    examples = materialize_example_arguments(
        artifact.example_arguments
        if representative_arguments is None
        else representative_arguments,
        device_ordinal=device_ordinal,
    )
    if artifact.kind == "optimizer":
        # Optimizer updates are intrinsically no-grad mutations.  Preserving a
        # Parameter example's requires-grad bit makes compile_fx introduce an
        # AOTAutograd mutation epilogue that is neither part of the optimizer
        # ABI nor valid for heterogeneous parameter shapes.
        examples = tuple(
            value.detach() if isinstance(value, torch.Tensor) else value
            for value in examples
        )
    graph_module = copy.deepcopy(artifact.graph_module)
    node_count = len(tuple(graph_module.graph.nodes))
    try:
        compiler: Any = compile_fx
        compiled: Callable[..., object] = compiler(graph_module, list(examples))
    except BaseException as exc:
        raise CaptureError(f"Inductor task compilation failed: {exc}") from exc
    return CompiledTask(
        artifact,
        compiled,
        examples,
        "torch-inductor",
        node_count,
    )


class CudaTaskProfiler:
    """Warm and measure compiled tasks through an installed ShadowSpill slab."""

    def __init__(
        self,
        library: Any,
        *,
        device_ordinal: int,
        warmup_iterations: int = 3,
        sample_iterations: int = 5,
        telemetry_capacity: int = 65_536,
    ) -> None:
        if warmup_iterations < 1:
            raise ValueError("task profiler requires at least one warmup")
        if sample_iterations < 1:
            raise ValueError("task profiler requires at least one sample")
        if telemetry_capacity < 1:
            raise ValueError("task profiler telemetry capacity must be positive")
        self._library = library
        self._device_ordinal = device_ordinal
        self._warmups = warmup_iterations
        self._samples = sample_iterations
        self._telemetry_capacity = telemetry_capacity
        self._next_task_id = 1 << 62
        self._executables: dict[str, CompiledTask] = {}

    def measure(self, artifact: ProfilableArtifact) -> TaskMeasurement:
        """Measure one compiled graph or bounded eager optimizer task."""

        if isinstance(artifact, OpaqueOptimizerArtifact):
            return self._measure_opaque_optimizer(artifact)
        if not isinstance(artifact, GraphArtifact):
            raise TypeError(f"unsupported profiling artifact {type(artifact).__name__}")

        executable = self._compiled(artifact)
        digest = artifact.compatibility_digest
        try:
            profiling_started = time.perf_counter_ns()
            measurement = self._measure_callable(
                executable,
                execution_provider=(
                    f"{executable.execution_provider}"
                    f"[fx_nodes={executable.graph_node_count}]"
                ),
            )
            measurement = replace(
                measurement,
                profiling_wall_time_ns=time.perf_counter_ns() - profiling_started,
            )
        except AllocationTelemetryError as exc:
            self._executables.pop(digest, None)
            raise AllocationTelemetryError(
                "allocator trace is incomplete for structural ABI "
                f"{digest} ({artifact.kind}; operators="
                f"{artifact.operator_targets}): {exc}"
            ) from exc
        except BaseException:
            self._executables.pop(digest, None)
            raise
        else:
            # The compiled function does not own its example arguments. Keeping
            # every unique ABI's CUDA examples alive until take_functions()
            # makes isolated profiling scale with the sum of model-stage
            # inputs, rather than the largest ABI. Retain only the executable.
            self._executables[digest] = CompiledTask(
                artifact,
                executable.function,
                (),
                executable.execution_provider,
                executable.graph_node_count,
            )
            return measurement
        finally:
            del executable

    def _measure_callable(
        self,
        executable: Callable[[], object],
        *,
        execution_provider: str = "bounded-eager",
    ) -> TaskMeasurement:
        """Measure a warmed no-argument task through the allocator boundary."""

        torch.cuda.set_device(self._device_ordinal)
        stream = torch.cuda.current_stream(self._device_ordinal)
        persistent_baseline = self._requested_allocated_bytes()
        persistent_high_water = persistent_baseline
        for _ in range(self._warmups):
            task_id = self._open_profile_task(stream)
            try:
                output = executable()
                del output
                self._close_profile_task(task_id, stream)
            except BaseException:
                self._library.shadowspill_pytorch_abort_task_range()
                raise
        stream.synchronize()
        self._library.shadowspill_pytorch_allocator_wait_idle()
        persistent_high_water = max(
            persistent_high_water, self._requested_allocated_bytes()
        )
        samples: list[int] = []
        event_factory: Any = torch.cuda.Event
        for _ in range(self._samples):
            start = event_factory(enable_timing=True)
            finish = event_factory(enable_timing=True)
            task_id = self._open_profile_task(stream)
            try:
                start.record(stream)
                output = executable()
                del output
                finish.record(stream)
                self._close_profile_task(task_id, stream)
            except BaseException:
                self._library.shadowspill_pytorch_abort_task_range()
                raise
            finish.synchronize()
            self._library.shadowspill_pytorch_allocator_wait_idle()
            samples.append(max(0, round(start.elapsed_time(finish) * 1_000_000)))
        self._library.shadowspill_pytorch_allocator_wait_idle()
        persistent_high_water = max(
            persistent_high_water, self._requested_allocated_bytes()
        )
        workspace, persistent_high_water = self._audit_workspace_retention(
            executable,
            stream,
            persistent_high_water=persistent_high_water,
        )
        fixed_bytes = max(0, persistent_high_water - persistent_baseline)
        # A shared provider cache may already be populated by another ABI.
        # Preserve at least this task's observed rotating live set so every
        # cached measurement remains independently conservative.
        fixed_bytes = max(fixed_bytes, sum(workspace.persistent_extent_bytes))
        fixed_extents = () if fixed_bytes == 0 else (fixed_bytes,)
        return TaskMeasurement(
            runtime_ns=round(statistics.median(samples)),
            workspace_requested_bytes=workspace.peak_requested_bytes,
            workspace_charged_bytes=workspace.peak_charged_bytes,
            workspace_extent_bytes=workspace.peak_extent_bytes,
            samples_ns=tuple(samples),
            provenance=(
                f"cuda-events+shadowspill-allocation-telemetry+{execution_provider}"
                + ("+bounded-retention-audit" if fixed_extents else "")
            ),
            allocation_trace=workspace.allocation_trace,
            persistent_extent_bytes=fixed_extents,
        )

    def _open_profile_task(self, stream: torch.cuda.Stream) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        status = int(
            self._library.shadowspill_pytorch_before_task(
                task_id, stream.cuda_stream, None, 0, None, 0
            )
        )
        if status != 0:
            raise CaptureError(f"profiling before_task failed with status {status}")
        return task_id

    def _close_profile_task(self, task_id: int, stream: torch.cuda.Stream) -> None:
        status = int(
            self._library.shadowspill_pytorch_after_task(
                task_id, stream.cuda_stream, None, 0, None, 0
            )
        )
        if status != 0:
            raise CaptureError(f"profiling after_task failed with status {status}")

    def _audit_workspace_retention(
        self,
        executable: Callable[[], object],
        stream: torch.cuda.Stream,
        *,
        persistent_high_water: int,
        maximum_iterations: int = 16,
    ) -> tuple[Any, int]:
        """Distinguish bounded provider caches from unbounded task leakage."""

        previous = self._requested_allocated_bytes()
        stable_observations = 0
        workspace: Any | None = None
        for _ in range(maximum_iterations):
            workspace = self._measure_workspace(executable, stream)
            current = self._requested_allocated_bytes()
            persistent_high_water = max(persistent_high_water, current)
            if not workspace.persistent_extent_bytes:
                return workspace, persistent_high_water
            if current == previous:
                stable_observations += 1
            else:
                stable_observations = 0
            if stable_observations >= 2:
                return workspace, persistent_high_water
            previous = current
        if workspace is None:
            raise AssertionError("workspace retention audit did not execute")
        raise AllocationTelemetryError(
            "task retains anonymous allocations without reaching a bounded "
            f"live-byte baseline after {maximum_iterations} invocations; "
            f"latest={workspace.persistent_extent_bytes}"
        )

    def _requested_allocated_bytes(self) -> int:
        return int(self._allocator_statistics().runtime.requested_allocated_bytes)

    def _allocator_statistics(self) -> AdapterStatistics:
        statistics = AdapterStatistics()
        status = int(
            self._library.shadowspill_pytorch_allocator_statistics(
                ctypes.byref(statistics)
            )
        )
        if status != 0:
            raise AllocationTelemetryError(
                f"allocator statistics failed during profiling with status {status}"
            )
        return statistics

    def _diagnose_allocator_idle(
        self,
        *,
        context: str,
    ) -> None:
        """Block on the runtime's progress-safe quiescence boundary."""

        status = int(self._library.shadowspill_pytorch_allocator_wait_idle())
        if status == 0:
            return
        detail = f"status={status}"
        if hasattr(self._library, "shadowspill_pytorch_allocator_statistics"):
            statistics = self._allocator_statistics().runtime
            detail = (
                f"{detail} pending={statistics.pending_retirements} "
                f"fenced={statistics.retirement_records_fenced} "
                f"evented={statistics.retirement_records_evented} "
                f"preparing={statistics.retirement_records_preparing} "
                f"unfenced={statistics.retirement_records_unfenced} "
                f"actions={statistics.queued_actions}"
            )
        raise AllocationTelemetryError(
            f"allocator failed to become idle during {context}: {detail}"
        )

    def _measure_opaque_optimizer(
        self, artifact: OpaqueOptimizerArtifact
    ) -> TaskMeasurement:
        optimizer = materialize_opaque_optimizer(
            artifact, device_ordinal=self._device_ordinal
        )

        def update(
            profiled_optimizer: torch.optim.Optimizer = optimizer,
        ) -> object:
            with torch.no_grad():
                return profiled_optimizer.step()

        measurement = self._measure_callable(
            update, execution_provider="opaque-optimizer"
        )
        del update
        del optimizer
        return measurement

    def take_functions(
        self,
        artifacts: Sequence[ProfilableArtifact],
        *,
        progress: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, Callable[..., object]]:
        """Transfer warmed unique entrypoints while releasing examples."""

        result: dict[str, Callable[..., object]] = {}
        stream: torch.cuda.Stream | None = None
        unique_count = len(
            {
                artifact.compatibility_digest
                for artifact in artifacts
                if not isinstance(artifact, OpaqueOptimizerArtifact)
            }
        )
        completed = 0
        for artifact in artifacts:
            if isinstance(artifact, OpaqueOptimizerArtifact):
                continue
            if not isinstance(artifact, GraphArtifact):
                raise TypeError(
                    f"unsupported executable artifact {type(artifact).__name__}"
                )
            digest = artifact.compatibility_digest
            if digest in result:
                continue
            executable = self._executables.pop(digest, None)
            completed += 1
            if progress is not None:
                progress(
                    completed,
                    unique_count,
                    "warmed" if executable is not None else "compiling",
                    digest,
                )
            if executable is None:
                executable = compile_artifact(
                    artifact, device_ordinal=self._device_ordinal
                )
                if stream is None:
                    stream = torch.cuda.current_stream(self._device_ordinal)
                for _ in range(self._warmups):
                    task_id = self._open_profile_task(stream)
                    try:
                        output = executable()
                        del output
                        self._close_profile_task(task_id, stream)
                    except BaseException:
                        self._library.shadowspill_pytorch_abort_task_range()
                        raise
                stream.synchronize()
                self._diagnose_allocator_idle(
                    context=f"compiled entrypoint {digest}",
                )
            result[digest] = executable.function
        return result

    def _compiled(self, artifact: GraphArtifact) -> CompiledTask:
        digest = artifact.compatibility_digest
        executable = self._executables.get(digest)
        if executable is None:
            executable = compile_artifact(artifact, device_ordinal=self._device_ordinal)
            self._executables[digest] = executable
        return executable

    def _measure_workspace(
        self, executable: Callable[[], object], stream: torch.cuda.Stream
    ) -> Any:
        task_id = self._next_task_id
        self._next_task_id += 1
        start_allocation_telemetry(self._library, capacity=self._telemetry_capacity)
        task_open = False
        output: object | None = None
        try:
            status = int(
                self._library.shadowspill_pytorch_before_task(
                    task_id, stream.cuda_stream, None, 0, None, 0
                )
            )
            if status != 0:
                raise CaptureError(f"profiling before_task failed with status {status}")
            task_open = True
            output = executable()
            output_allocations = self._output_allocation_views(output)
            # Profiling does not retain task results. Release them while the
            # task range is still active so output-dependent temporary frees
            # remain attributable to this ABI. The allocator retires their
            # physical ranges against the active compute stream.
            output = None
            status = int(
                self._library.shadowspill_pytorch_after_task(
                    task_id, stream.cuda_stream, None, 0, None, 0
                )
            )
            task_open = False
            if status != 0:
                raise CaptureError(f"profiling after_task failed with status {status}")
            stream.synchronize()
            self._library.shadowspill_pytorch_allocator_wait_idle()
        except BaseException:
            if task_open:
                self._library.shadowspill_pytorch_abort_task_range()
            raise
        finally:
            stop_allocation_telemetry(self._library)
        events = read_allocation_telemetry(self._library)
        return summarize_task_workspace(
            events,
            task_id=task_id,
            output_allocation_views=output_allocations,
        )

    def _output_allocation_views(
        self, output: object
    ) -> dict[int, tuple[tuple[int, int], ...]]:

        views_by_allocation: dict[int, list[tuple[int, int]]] = {}
        leaves, _ = tree_flatten(output)
        for leaf_index, leaf in enumerate(leaves):
            if not isinstance(leaf, torch.Tensor) or not leaf.is_cuda:
                continue
            address = leaf.untyped_storage().data_ptr()
            if address == 0:
                continue
            allocation = Allocation()
            status = int(
                self._library.shadowspill_pytorch_allocation_for_pointer(
                    address, ctypes.byref(allocation)
                )
            )
            if status != 0:
                raise CaptureError(
                    "compiled task returned storage outside the ShadowSpill slab"
                )
            allocation_pointer = int(allocation.pointer or 0)
            view_pointer = int(leaf.data_ptr())
            offset_bytes = view_pointer - allocation_pointer
            if offset_bytes < 0 or offset_bytes > int(allocation.requested_bytes):
                raise CaptureError(
                    "compiled output view lies outside its allocator record"
                )
            views_by_allocation.setdefault(allocation.allocation_id, []).append(
                (leaf_index, offset_bytes)
            )
        return {
            allocation_id: tuple(views)
            for allocation_id, views in views_by_allocation.items()
        }
