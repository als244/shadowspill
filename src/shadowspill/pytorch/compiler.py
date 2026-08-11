"""Compilation and isolated CUDA measurement for one structural task ABI."""

from __future__ import annotations

import copy
import ctypes
import gc
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch._inductor.compile_fx import compile_fx
from torch.utils._pytree import tree_flatten

from ._abi import Allocation
from ._telemetry import (
    read_allocation_telemetry,
    start_allocation_telemetry,
    stop_allocation_telemetry,
    summarize_task_workspace,
)
from .capture import GraphArtifact
from .contracts import CaptureError
from .profiling import ProfileEnvironment, TaskMeasurement


@dataclass(frozen=True, slots=True)
class CompiledTask:
    """One compiled graph and allocator-owned representative arguments."""

    artifact: GraphArtifact
    function: Callable[..., object]
    example_arguments: tuple[object, ...]

    def __call__(self) -> object:
        context = torch.no_grad() if self.artifact.kind == "optimizer" else _null()
        with context:
            return self.function(*self.example_arguments)


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        del exc


def profile_environment(*, device_ordinal: int, provider_id: str) -> ProfileEnvironment:
    """Describe every implementation attribute that can change task cost."""

    properties = torch.cuda.get_device_properties(device_ordinal)
    return ProfileEnvironment(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device_name=properties.name,
        compute_capability=(properties.major, properties.minor),
        compiler_id="torch-inductor-compile-fx",
        provider_id=provider_id,
    )


def materialize_example_arguments(
    arguments: Sequence[object], *, device_ordinal: int
) -> tuple[object, ...]:
    """Create real CUDA values while preserving tensor storage aliases."""

    device = torch.device("cuda", device_ordinal)
    storages: dict[int, torch.Tensor] = {}
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
            key = source_storage._cdata
            storage = storages.get(key)
            if storage is None:
                storage = torch.empty(
                    source_storage.nbytes(), dtype=torch.uint8, device=device
                )
                storage.zero_()
                storages[key] = storage
            tensor = torch.empty(0, dtype=argument.dtype, device=device)
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
    try:
        graph_module = copy.deepcopy(artifact.graph_module)
        compiler: Any = compile_fx
        compiled: Callable[..., object] = compiler(graph_module, list(examples))
    except BaseException as exc:
        raise CaptureError(f"Inductor task compilation failed: {exc}") from exc
    return CompiledTask(artifact, compiled, examples)


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

    def measure(self, artifact: GraphArtifact) -> TaskMeasurement:
        """Compile once, then measure calibrated time and one exact live set."""

        executable = compile_artifact(artifact, device_ordinal=self._device_ordinal)
        torch.cuda.set_device(self._device_ordinal)
        stream = torch.cuda.current_stream(self._device_ordinal)
        for _ in range(self._warmups):
            self._discard_output(executable())
        stream.synchronize()
        samples: list[int] = []
        event_factory: Any = torch.cuda.Event
        for _ in range(self._samples):
            start = event_factory(enable_timing=True)
            finish = event_factory(enable_timing=True)
            start.record(stream)
            output = executable()
            finish.record(stream)
            finish.synchronize()
            samples.append(max(0, round(start.elapsed_time(finish) * 1_000_000)))
            del output
            gc.collect()
        workspace = self._measure_workspace(executable, stream)
        return TaskMeasurement(
            runtime_ns=round(statistics.median(samples)),
            workspace_requested_bytes=workspace.peak_requested_bytes,
            workspace_charged_bytes=workspace.peak_charged_bytes,
            workspace_extent_bytes=workspace.peak_extent_bytes,
            samples_ns=tuple(samples),
            provenance="cuda-events+shadowspill-allocation-telemetry",
        )

    def _measure_workspace(
        self, executable: CompiledTask, stream: torch.cuda.Stream
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
            status = int(
                self._library.shadowspill_pytorch_after_task(
                    task_id, stream.cuda_stream, None, 0, None, 0
                )
            )
            task_open = False
            if status != 0:
                raise CaptureError(f"profiling after_task failed with status {status}")
            stream.synchronize()
            output_ids = self._output_allocation_ids(output)
        except BaseException:
            if task_open:
                self._library.shadowspill_pytorch_abort_task_range()
            raise
        finally:
            stop_allocation_telemetry(self._library)
        events = read_allocation_telemetry(self._library)
        self._discard_output(output)
        return summarize_task_workspace(
            events,
            task_id=task_id,
            output_allocation_ids=output_ids,
        )

    def _output_allocation_ids(self, output: object) -> tuple[int, ...]:
        allocation_ids: set[int] = set()
        leaves, _ = tree_flatten(output)
        for leaf in leaves:
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
            allocation_ids.add(allocation.allocation_id)
        return tuple(sorted(allocation_ids))

    @staticmethod
    def _discard_output(output: object) -> None:
        del output
        gc.collect()
