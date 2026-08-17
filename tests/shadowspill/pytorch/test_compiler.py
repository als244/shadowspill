from __future__ import annotations

import ctypes
import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation import compiler as compiler_module
from shadowspill.pytorch.compilation import inductor as inductor_module
from shadowspill.pytorch.compilation.compiler import (
    CompiledTask,
    compile_artifact,
    materialize_example_arguments,
)
from shadowspill.pytorch.compilation.inductor import (
    ExecutableRootAllocation,
    ExecutableTaskManifest,
    compile_explicit_inductor_task,
    compile_inductor_task,
)
from shadowspill.pytorch.contracts import (
    CaptureError,
    CompilationError,
    ProfilingError,
)
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.profiling import (
    TaskMeasurement,
    profile_environment,
)
from shadowspill.pytorch.profiling import profiler as profiler_module
from shadowspill.pytorch.profiling.inputs import (
    materialize_representative_inputs,
)
from shadowspill.pytorch.profiling.profiler import CudaTaskProfiler
from shadowspill.pytorch.runtime_adapter.abi import Allocation
from shadowspill.pytorch.runtime_adapter.telemetry import AllocationTelemetryError


class _Add(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


class _MultiplyByOne(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.ones_like(value)


def _artifact(kind: str = "inference") -> GraphArtifact:
    inputs = (torch.randn(8, 8), torch.randn(8, 8))
    return GraphArtifact.capture(
        kind=kind,  # type: ignore[arg-type]
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=inputs,
    )


def _manifest(artifact: GraphArtifact) -> ExecutableTaskManifest:
    return ExecutableTaskManifest(
        semantic_contract_digest=artifact.storage_contract.compatibility_digest,
        storage_contract=artifact.storage_contract,
        contract_capture_ns=0,
        compatibility_digest="0" * 64,
        root_allocations=tuple(
            ExecutableRootAllocation(
                root.root_id,
                0 if root.kind.value == "input" else root.minimum_span_bytes,
            )
            for root in artifact.storage_contract.roots
        ),
    )


def _compiled_task(
    artifact: GraphArtifact,
    function: Any,
    arguments: tuple[object, ...] = (),
) -> CompiledTask:
    return CompiledTask(artifact, function, arguments, _manifest(artifact))


def test_compiled_task_disables_dispatcher_autograd() -> None:
    observed: list[bool] = []
    artifact = _artifact("forward")
    executable = _compiled_task(
        artifact,
        lambda: observed.append(torch.is_grad_enabled()),
    )

    with torch.enable_grad():
        executable()

    assert observed == [False]


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
    assert tuple(
        item.requested_bytes for item in executable.manifest.root_allocations
    ) == (256,)
    output = executable()
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(
        output,
        executable.example_arguments[0] + executable.example_arguments[1],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_cache_restores_the_exact_executable_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    artifact = _artifact()

    first = compile_artifact(artifact, device_ordinal=0)
    first_output = first()
    monkeypatch.setattr(
        inductor_module,
        "_graph_lowering_contract",
        lambda *args, **kwargs: pytest.fail(
            "warm AOT/Inductor cache unexpectedly rebuilt GraphLowering"
        ),
    )
    second = compile_artifact(artifact, device_ordinal=0)
    second_output = second()

    torch.testing.assert_close(first_output, second_output)
    assert first.manifest.compatibility_digest == second.manifest.compatibility_digest
    assert first.manifest.storage_contract == second.manifest.storage_contract
    assert tuple((tmp_path / "shadowspill" / "task_manifests").rglob("*.json"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_manifest_captures_post_grad_output_alias() -> None:
    value = torch.randn(32)
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(_MultiplyByOne()),
        example_inputs=(value,),
    )
    assert artifact.storage_contract.roots[0].kind.value == "fresh"

    executable = compile_artifact(artifact, device_ordinal=0)
    compiled_contract = executable.manifest.storage_contract
    assert compiled_contract.roots[0].kind.value == "input"
    assert compiled_contract.roots[0].source_input == 0
    output = executable()
    assert isinstance(output, torch.Tensor)
    assert output.data_ptr() == executable.example_arguments[0].data_ptr()  # type: ignore[union-attr]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("module", [_Add(), _MultiplyByOne()])
def test_explicit_inductor_path_matches_outer_aot_for_inference(
    module: nn.Module,
) -> None:
    inputs = (
        (torch.randn(8, 8), torch.randn(8, 8))
        if isinstance(module, _Add)
        else (torch.randn(8, 8),)
    )
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(module),
        example_inputs=inputs,
    )
    arguments = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in materialize_example_arguments(inputs, device_ordinal=0)
    )

    outer = compile_inductor_task(
        artifact.graph_module,
        arguments,
        semantic_contract=artifact.storage_contract,
    )
    explicit = compile_explicit_inductor_task(
        artifact.graph_module,
        arguments,
        semantic_contract=artifact.storage_contract,
    )

    torch.testing.assert_close(
        explicit.function(*arguments),
        outer.function(*arguments),
    )
    assert explicit.manifest.storage_contract == outer.manifest.storage_contract
    assert explicit.manifest.root_allocations == outer.manifest.root_allocations
    phase_names = {name for name, _duration in explicit.phase_timings_ns}
    assert "torch_decomposition_normalization" in phase_names
    assert "torch_inductor_core" in phase_names
    assert all(duration > 0 for _name, duration in explicit.phase_timings_ns)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_manifest_hydration_restores_arguments_before_measurement() -> None:
    artifact = _artifact()
    profiler = CudaTaskProfiler(
        _TaskLibrary(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    profiler.prepare_manifests((artifact,))
    observed_arguments: list[int] = []

    def measure(executable: Any, **_options: object) -> Any:
        observed_arguments.append(len(executable.example_arguments))
        return TaskMeasurement(1, 0, 0, (), (1,), "test")

    profiler._measure_callable = measure  # type: ignore[method-assign]

    profiler.measure(artifact)

    assert observed_arguments == [len(artifact.example_arguments)]
    assert profiler._compiled(artifact).example_arguments == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_optimizer_compilation_uses_no_grad_mutation_abi() -> None:
    model = nn.Sequential(nn.Linear(6, 10), nn.Linear(10, 3))
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    captured = capture_optimizer(dict(model.named_parameters()), optimizer)
    assert captured.recurrent is not None

    executable = compile_artifact(captured.recurrent, device_ordinal=0)
    representatives = materialize_representative_inputs(
        captured.recurrent, device_ordinal=0
    )
    with torch.no_grad():
        outputs = executable.function(*representatives.arguments)
    assert isinstance(outputs, tuple | list)
    assert len(outputs) == len(captured.mutation_names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_explicit_optimizer_preserves_outer_aot_mutation_abi() -> None:
    model = nn.Sequential(nn.Linear(6, 10), nn.Linear(10, 3))
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    artifact = capture_optimizer(dict(model.named_parameters()), optimizer).recurrent
    assert artifact is not None

    outer_arguments = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in materialize_example_arguments(
            artifact.example_arguments, device_ordinal=0
        )
    )
    explicit_arguments = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in materialize_example_arguments(
            artifact.example_arguments, device_ordinal=0
        )
    )
    outer = compile_inductor_task(
        artifact.graph_module,
        outer_arguments,
        semantic_contract=artifact.storage_contract,
    )
    explicit = compile_explicit_inductor_task(
        artifact.graph_module,
        explicit_arguments,
        semantic_contract=artifact.storage_contract,
    )

    outer_outputs = outer.function(*outer_arguments)
    explicit_outputs = explicit.function(*explicit_arguments)
    torch.testing.assert_close(explicit_outputs, outer_outputs)
    torch.testing.assert_close(explicit_arguments, outer_arguments)
    assert explicit.manifest.storage_contract == outer.manifest.storage_contract
    assert explicit.manifest.root_allocations == outer.manifest.root_allocations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_measurement_uses_events_and_reports_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _TaskLibrary()
    profiler = CudaTaskProfiler(
        library, device_ordinal=0, warmup_iterations=1, sample_iterations=2
    )
    workspace = SimpleNamespace(
        peak_requested_bytes=64,
        peak_charged_bytes=256,
        peak_extent_bytes=(256,),
        allocation_trace=(),
        allocation_contract_trace=(),
        output_input_bindings=(),
        persistent_allocation_ids=(),
        persistent_extent_bytes=(),
    )
    monkeypatch.setattr(profiler, "_measure_workspace", lambda task, stream: workspace)
    monkeypatch.setattr(profiler, "_requested_allocated_bytes", lambda: 0)
    measurement = profiler.measure(_artifact())
    assert measurement.runtime_ns >= 0
    assert len(measurement.samples_ns) == 2
    assert measurement.workspace_charged_bytes == 256
    assert measurement.provenance.startswith("cuda-events")
    assert "+torch-inductor" in measurement.provenance

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

    @staticmethod
    def shadowspill_pytorch_allocator_wait_idle() -> int:
        return 0

    @staticmethod
    def shadowspill_pytorch_allocator_failure(*arguments: object) -> int:
        del arguments
        return 0


class _Stream:
    cuda_stream = 17

    def synchronize(self) -> None:
        return None


def test_workspace_boundary_always_stops_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    sentinel = object()
    monkeypatch.setattr(
        profiler_module,
        "start_allocation_telemetry",
        lambda library, capacity: calls.append(f"start:{capacity}"),
    )
    monkeypatch.setattr(
        profiler_module,
        "stop_allocation_telemetry",
        lambda library: calls.append("stop"),
    )
    monkeypatch.setattr(
        profiler_module, "read_allocation_telemetry", lambda library: ()
    )
    monkeypatch.setattr(
        profiler_module,
        "summarize_task_workspace",
        lambda events, **options: sentinel,
    )
    library = _TaskLibrary()
    profiler = CudaTaskProfiler(
        library, device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    artifact = _artifact()
    executable = _compiled_task(artifact, lambda *args: torch.ones(1))
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


def test_workspace_releases_disposable_results_before_after_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Result:
        def __del__(self) -> None:
            calls.append("release-result")

    class Library(_TaskLibrary):
        def shadowspill_pytorch_after_task(self, *arguments: object) -> int:
            del arguments
            calls.append("after-task")
            return 0

    monkeypatch.setattr(
        profiler_module, "start_allocation_telemetry", lambda *a, **k: None
    )
    monkeypatch.setattr(
        profiler_module, "stop_allocation_telemetry", lambda *a, **k: None
    )
    monkeypatch.setattr(
        profiler_module, "read_allocation_telemetry", lambda library: ()
    )
    monkeypatch.setattr(
        profiler_module,
        "summarize_task_workspace",
        lambda events, **options: object(),
    )
    profiler = CudaTaskProfiler(
        Library(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    artifact = _artifact()
    executable = _compiled_task(artifact, lambda: Result())
    profiler._measure_workspace(executable, _Stream())  # type: ignore[arg-type]
    assert calls == ["release-result", "after-task"]


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
            allocation.pointer = address
            allocation.requested_bytes = 16
            allocation.charged_bytes = 16
            return 0

    profiler = CudaTaskProfiler(
        _Lookup(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    tensor = torch.empty(4, device="cuda")
    assert profiler._output_allocation_views((tensor, tensor.view(2, 2))) == (
        {91: ((0, 0), (1, 0))},
        (),
    )

    class _Missing:
        @staticmethod
        def shadowspill_pytorch_allocation_for_pointer(*arguments: object) -> int:
            del arguments
            return 5

    missing = CudaTaskProfiler(
        _Missing(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    with pytest.raises(CaptureError, match="outside"):
        missing._output_allocation_views(tensor)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"warmup_iterations": 0}, "warmup"),
        ({"sample_iterations": 0}, "sample"),
        ({"telemetry_capacity": 0}, "capacity"),
        ({"allocation_probe_seeds": 0}, "allocation paths"),
        ({"allocation_probe_repetitions": 1}, "allocation paths"),
    ],
)
def test_profiler_rejects_empty_calibration(
    options: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CudaTaskProfiler(object(), device_ordinal=0, **options)


def test_retention_audit_accepts_a_stable_live_byte_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = CudaTaskProfiler(
        object(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    workspace = SimpleNamespace(persistent_extent_bytes=(32,))
    measurements = iter((100, 132, 132, 132))
    monkeypatch.setattr(
        profiler,
        "_requested_allocated_bytes",
        lambda: next(measurements),
    )
    monkeypatch.setattr(
        profiler, "_measure_workspace", lambda executable, stream: workspace
    )
    captured, high_water = profiler._audit_workspace_retention(
        lambda: None,
        object(),  # type: ignore[arg-type]
        persistent_high_water=100,
    )
    assert captured is workspace
    assert high_water == 132


def test_retention_audit_rejects_unbounded_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = CudaTaskProfiler(
        object(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    workspace = SimpleNamespace(persistent_extent_bytes=(32,))
    measurements = iter((100, 132, 164, 196))
    monkeypatch.setattr(
        profiler,
        "_requested_allocated_bytes",
        lambda: next(measurements),
    )
    monkeypatch.setattr(
        profiler, "_measure_workspace", lambda executable, stream: workspace
    )
    with pytest.raises(AllocationTelemetryError, match="without reaching"):
        profiler._audit_workspace_retention(
            lambda: None,
            object(),  # type: ignore[arg-type]
            persistent_high_water=100,
            maximum_iterations=3,
        )


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
        profiler.take_compiled_tasks((artifact,))


def test_compiler_function_transfer_deduplicates_structural_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact()
    calls: list[str] = []

    def compile_once(value: GraphArtifact, *, device_ordinal: int) -> CompiledTask:
        calls.append(value.compatibility_digest)
        assert device_ordinal == 0
        return _compiled_task(value, lambda *arguments: arguments)

    monkeypatch.setattr(compiler_module, "compile_artifact", compile_once)
    library = _TaskLibrary()
    profiler = CudaTaskProfiler(
        library, device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    compiled = profiler.take_compiled_tasks((artifact, artifact))
    assert tuple(compiled.functions) == (artifact.compatibility_digest,)
    assert tuple(compiled.manifests) == (artifact.compatibility_digest,)
    assert calls == [artifact.compatibility_digest]

    profiler._compiled(artifact)
    profiler._compiled(artifact)
    assert calls == [artifact.compatibility_digest, artifact.compatibility_digest]


def test_compiler_failure_has_structural_context_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact()

    def fail_compile(value: GraphArtifact, *, device_ordinal: int) -> CompiledTask:
        del value, device_ordinal
        raise RuntimeError("compiler exploded")

    monkeypatch.setattr(compiler_module, "compile_artifact", fail_compile)
    profiler = CudaTaskProfiler(
        object(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )

    with pytest.raises(CompilationError, match="compiler exploded") as captured:
        profiler._compiled(artifact)

    assert captured.value.structural_abi == artifact.compatibility_digest
    assert captured.value.task_kind == artifact.kind
    assert captured.value.operators == artifact.operator_targets
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_profile_failure_has_structural_context_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact()
    arguments = (torch.ones(1),)

    monkeypatch.setattr(
        compiler_module,
        "compile_artifact",
        lambda value, *, device_ordinal: _compiled_task(
            value,
            lambda *items: items,
            arguments,
        ),
    )
    profiler = CudaTaskProfiler(
        object(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    monkeypatch.setattr(
        profiler,
        "_measure_callable",
        lambda *arguments, **options: (_ for _ in ()).throw(
            RuntimeError("kernel exploded")
        ),
    )

    with pytest.raises(ProfilingError, match="kernel exploded") as captured:
        profiler.measure(artifact)

    assert captured.value.structural_abi == artifact.compatibility_digest
    assert captured.value.task_kind == artifact.kind
    assert captured.value.operators == artifact.operator_targets
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_measurement_releases_cuda_examples_between_structural_abis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.pytorch.profiling import TaskMeasurement

    artifact = _artifact()
    examples = [torch.ones(8)]
    example_reference = weakref.ref(examples[0])

    def compile_with_large_example(
        value: GraphArtifact, *, device_ordinal: int
    ) -> CompiledTask:
        assert device_ordinal == 0
        return _compiled_task(
            value,
            lambda *arguments: arguments,
            (examples[0],),
        )

    measurement = TaskMeasurement(1, 0, 0, (), (1,), "test")
    monkeypatch.setattr(compiler_module, "compile_artifact", compile_with_large_example)
    profiler = CudaTaskProfiler(
        object(), device_ordinal=0, warmup_iterations=1, sample_iterations=1
    )
    stale_frames: list[object] = []

    def measure_and_retain_wrapper(executable: object, **options: object) -> object:
        del options
        stale_frames.append(executable)
        return measurement

    monkeypatch.setattr(
        profiler,
        "_measure_callable",
        measure_and_retain_wrapper,
    )

    observed = profiler.measure(artifact)
    assert observed.runtime_ns == measurement.runtime_ns
    assert observed.workspace_charged_bytes == measurement.workspace_charged_bytes
    assert observed.profiling_wall_time_ns > 0
    examples.clear()
    gc.collect()
    assert example_reference() is None
    assert stale_frames
    assert not stale_frames[0].example_arguments  # type: ignore[attr-defined]
    assert profiler._compiled(artifact).example_arguments == ()
