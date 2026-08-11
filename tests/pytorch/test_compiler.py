from __future__ import annotations

import ctypes
import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch._abi import Allocation
from shadowspill.pytorch.capture import GraphArtifact
from shadowspill.pytorch.compiler import (
    CompiledTask,
    CudaTaskProfiler,
    compile_artifact,
    materialize_example_arguments,
    profile_environment,
)
from shadowspill.pytorch.contracts import CaptureError
from shadowspill.pytorch.optimizer import capture_optimizer


class _Add(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


def _artifact(kind: str = "inference") -> GraphArtifact:
    inputs = (torch.randn(8, 8), torch.randn(8, 8))
    return GraphArtifact.capture(
        kind=kind,  # type: ignore[arg-type]
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=inputs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_materialization_preserves_storage_alias_and_compiles() -> None:
    source = torch.arange(32, dtype=torch.float32)
    first = source[2:18].view(4, 4)
    second = source[4:20].view(4, 4)
    arguments = materialize_example_arguments(
        (first, second, {"mode": 2}), device_ordinal=0
    )
    left, right, metadata = arguments
    assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
    assert left.untyped_storage()._cdata == right.untyped_storage()._cdata
    assert left.storage_offset() == 2
    assert right.storage_offset() == 4
    assert metadata == {"mode": 2}

    executable = compile_artifact(_artifact(), device_ordinal=0)
    output = executable()
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_optimizer_compilation_uses_no_grad_mutation_abi() -> None:
    model = nn.Sequential(nn.Linear(6, 10), nn.Linear(10, 3))
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    captured = capture_optimizer(dict(model.named_parameters()), optimizer)
    assert captured.recurrent is not None

    executable = compile_artifact(captured.recurrent, device_ordinal=0)
    outputs = executable()
    assert isinstance(outputs, (tuple, list))
    assert len(outputs) == len(captured.mutation_names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_measurement_uses_events_and_reports_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = CudaTaskProfiler(
        object(), device_ordinal=0, warmup_iterations=1, sample_iterations=2
    )
    workspace = SimpleNamespace(
        peak_requested_bytes=64,
        peak_charged_bytes=256,
        peak_extent_bytes=(256,),
    )
    monkeypatch.setattr(profiler, "_measure_workspace", lambda task, stream: workspace)
    measurement = profiler.measure(_artifact())
    assert measurement.runtime_ns >= 0
    assert len(measurement.samples_ns) == 2
    assert measurement.workspace_charged_bytes == 256
    assert measurement.provenance.startswith("cuda-events")

    environment = profile_environment(device_ordinal=0, provider_id="test")
    assert environment.compute_capability == torch.cuda.get_device_capability(0)
    assert environment.provider_id == "test"


class _TaskLibrary:
    def __init__(self, *, before_status: int = 0) -> None:
        self.before_status = before_status
        self.aborted = False

    def shadowspill_pytorch_before_task(self, *arguments: object) -> int:
        del arguments
        return self.before_status

    def shadowspill_pytorch_after_task(self, *arguments: object) -> int:
        del arguments
        return 0

    def shadowspill_pytorch_abort_task_range(self) -> None:
        self.aborted = True


class _Stream:
    cuda_stream = 17

    def synchronize(self) -> None:
        return None


def test_workspace_boundary_always_stops_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.pytorch import compiler as compiler_module

    calls: list[str] = []
    sentinel = object()
    monkeypatch.setattr(
        compiler_module,
        "start_allocation_telemetry",
        lambda library, capacity: calls.append(f"start:{capacity}"),
    )
    monkeypatch.setattr(
        compiler_module,
        "stop_allocation_telemetry",
        lambda library: calls.append("stop"),
    )
    monkeypatch.setattr(
        compiler_module, "read_allocation_telemetry", lambda library: ()
    )
    monkeypatch.setattr(
        compiler_module,
        "summarize_task_workspace",
        lambda events, **options: sentinel,
    )
    library = _TaskLibrary()
    profiler = CudaTaskProfiler(
        library, device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    executable = CompiledTask(_artifact(), lambda *args: torch.ones(1), ())
    assert profiler._measure_workspace(executable, _Stream()) is sentinel  # type: ignore[arg-type]
    assert calls == ["start:65536", "stop"]

    failing_library = _TaskLibrary(before_status=5)
    failing = CudaTaskProfiler(
        failing_library,
        device_ordinal=0,
        warmup_iterations=1,
        sample_iterations=1,
    )
    with pytest.raises(CaptureError, match="before_task"):
        failing._measure_workspace(executable, _Stream())  # type: ignore[arg-type]
    assert calls[-2:] == ["start:65536", "stop"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_output_allocation_lookup_is_exact() -> None:
    class _Lookup:
        @staticmethod
        def shadowspill_pytorch_allocation_for_pointer(
            address: int, allocation_pointer: Any
        ) -> int:
            assert address != 0
            allocation = ctypes.cast(allocation_pointer, ctypes.POINTER(Allocation))[0]
            allocation.allocation_id = 91
            return 0

    profiler = CudaTaskProfiler(
        _Lookup(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    tensor = torch.empty(4, device="cuda")
    assert profiler._output_allocation_ids((tensor, tensor.view(2, 2))) == (91,)

    class _Missing:
        @staticmethod
        def shadowspill_pytorch_allocation_for_pointer(*arguments: object) -> int:
            del arguments
            return 5

    missing = CudaTaskProfiler(
        _Missing(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    with pytest.raises(CaptureError, match="outside"):
        missing._output_allocation_ids(tensor)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"warmup_iterations": 0}, "warmup"),
        ({"sample_iterations": 0}, "sample"),
        ({"telemetry_capacity": 0}, "capacity"),
    ],
)
def test_profiler_rejects_empty_calibration(
    options: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CudaTaskProfiler(object(), device_ordinal=0, **options)


def test_profiler_rejects_unknown_artifact_protocol() -> None:
    class _Unknown:
        compatibility_digest = "unknown"

    profiler = CudaTaskProfiler(
        object(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    artifact = _Unknown()
    with pytest.raises(TypeError, match="unsupported profiling artifact"):
        profiler.measure(artifact)
    with pytest.raises(TypeError, match="unsupported executable artifact"):
        profiler.take_functions((artifact,))


def test_compiler_function_transfer_deduplicates_structural_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.pytorch import compiler as compiler_module

    artifact = _artifact()
    calls: list[str] = []

    def compile_once(value: GraphArtifact, *, device_ordinal: int) -> CompiledTask:
        calls.append(value.compatibility_digest)
        assert device_ordinal == 0
        return CompiledTask(value, lambda *arguments: arguments, ())

    monkeypatch.setattr(compiler_module, "compile_artifact", compile_once)
    profiler = CudaTaskProfiler(
        object(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    functions = profiler.take_functions((artifact, artifact))
    assert tuple(functions) == (artifact.compatibility_digest,)
    assert calls == [artifact.compatibility_digest]

    profiler._compiled(artifact)
    profiler._compiled(artifact)
    assert calls == [artifact.compatibility_digest, artifact.compatibility_digest]


def test_measurement_releases_cuda_examples_between_structural_abis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.pytorch import compiler as compiler_module
    from shadowspill.pytorch.profiling import TaskMeasurement

    artifact = _artifact()
    examples = [torch.ones(8)]
    example_reference = weakref.ref(examples[0])

    def compile_with_large_example(
        value: GraphArtifact, *, device_ordinal: int
    ) -> CompiledTask:
        assert device_ordinal == 0
        return CompiledTask(value, lambda *arguments: arguments, (examples[0],))

    measurement = TaskMeasurement(1, 0, 0, (), (1,), "test")
    monkeypatch.setattr(compiler_module, "compile_artifact", compile_with_large_example)
    profiler = CudaTaskProfiler(
        object(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    monkeypatch.setattr(profiler, "_measure_callable", lambda executable: measurement)

    assert profiler.measure(artifact) is measurement
    examples.clear()
    gc.collect()
    assert example_reference() is None
    assert profiler._compiled(artifact).example_arguments == ()
