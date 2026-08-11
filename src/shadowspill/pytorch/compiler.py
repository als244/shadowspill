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
from .optimizer import OpaqueOptimizerArtifact, materialize_opaque_optimizer
from .profiling import ProfilableArtifact, ProfileEnvironment, TaskMeasurement


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
    if artifact.kind == "optimizer":
        # Optimizer updates are intrinsically no-grad mutations.  Preserving a
        # Parameter example's requires-grad bit makes compile_fx introduce an
        # AOTAutograd mutation epilogue that is neither part of the optimizer
        # ABI nor valid for heterogeneous parameter shapes.
        examples = tuple(
            value.detach() if isinstance(value, torch.Tensor) else value
            for value in examples
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
        self._executables: dict[str, CompiledTask] = {}

    def measure(self, artifact: ProfilableArtifact) -> TaskMeasurement:
        """Measure one compiled graph or bounded eager optimizer task."""

        if isinstance(artifact, OpaqueOptimizerArtifact):
            return self._measure_opaque_optimizer(artifact)
        if not isinstance(artifact, GraphArtifact):
            raise TypeError(f"unsupported profiling artifact {type(artifact).__name__}")

        executable = self._compiled(artifact)
        return self._measure_callable(executable)

    def _measure_callable(self, executable: Callable[[], object]) -> TaskMeasurement:
        """Measure a warmed no-argument task through the allocator boundary."""

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

        measurement = self._measure_callable(update)
        del update
        del optimizer
        gc.collect()
        return measurement

    def take_functions(
        self, artifacts: Sequence[ProfilableArtifact]
    ) -> dict[str, Callable[..., object]]:
        """Transfer unique compiled entrypoints while releasing examples."""

        result: dict[str, Callable[..., object]] = {}
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
            if executable is None:
                executable = compile_artifact(
                    artifact, device_ordinal=self._device_ordinal
                )
            result[digest] = executable.function
        gc.collect()
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
