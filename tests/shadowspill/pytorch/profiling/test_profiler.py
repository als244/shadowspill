from __future__ import annotations

import ctypes
import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from shadowspill.errors import CaptureError, CompilationError, ProfilingError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation import compiler as compiler_module
from shadowspill.pytorch.compilation.compiler import CompiledTask
from shadowspill.pytorch.profiling import (
    TaskMeasurement,
    profile_environment,
)
from shadowspill.pytorch.profiling import profiler as profiler_package
from shadowspill.pytorch.profiling.profiler import TaskProfiler
from shadowspill.pytorch.profiling.profiler import measurement as measurement_module
from shadowspill.pytorch.profiling.profiler import workspace as workspace_module
from shadowspill.pytorch.profiling.profiler.boundary import AllocatorBoundary
from shadowspill.pytorch.profiling.profiler.workspace import (
    WorkspaceObservation,
    WorkspaceTimings,
    audit_workspace_retention,
    measure_workspace,
    output_allocation_views,
)
from shadowspill.runtime import failures as failures_module
from shadowspill.runtime.abi import Allocation
from shadowspill.runtime.telemetry import AllocationTelemetryError
from shadowspill.task.manifest import ExecutableRootAllocation, ExecutableTaskManifest
from tests.shadowspill.runtime._timing import TimingLibrary, install


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


def _compiled_task(
    artifact: GraphArtifact,
    function: Any,
    arguments: tuple[object, ...] = (),
) -> CompiledTask:
    manifest = ExecutableTaskManifest(
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
    return CompiledTask(artifact, function, arguments, manifest)


@pytest.fixture(autouse=True)
def _timing(monkeypatch: pytest.MonkeyPatch) -> TimingLibrary:
    """Every profiler here times against the runtime, so stand in for it."""

    return install(monkeypatch, TimingLibrary(tick_ms=1.0))


def _profiler(library: Any = None, **options: int) -> TaskProfiler:
    return TaskProfiler(
        object() if library is None else library,
        runtime_handle=0,
        plan_id=1,
        device_ordinal=0,
        warmup_iterations=options.pop("warmup_iterations", 1),
        sample_iterations=options.pop("sample_iterations", 1),
        **options,
    )


class _IdleRuntime:
    """The neutral runtime as a fake library whose drain always succeeds."""

    @staticmethod
    def shadowspill_runtime_wait_idle(runtime_handle: int) -> int:
        del runtime_handle
        return 0


class _TaskLibrary:
    def __init__(self, *, before_status: int = 0) -> None:
        self.before_status = before_status
        self.aborted = False

    def shadowspill_pytorch_allocation_scope_begin(self, *arguments: object) -> int:
        del arguments
        return self.before_status

    def shadowspill_pytorch_allocation_scope_end(self, *arguments: object) -> int:
        del arguments
        return 0

    def shadowspill_pytorch_allocation_scope_abort(self) -> None:
        self.aborted = True

    @staticmethod
    def shadowspill_pytorch_allocator_statistics(*arguments: object) -> int:
        # The caller passes zeroed storage, so reporting success leaves the
        # allocation-event overflow flag clear: this double records nothing.
        return 0

    @staticmethod
    def shadowspill_pytorch_allocator_failure(*arguments: object) -> int:
        del arguments
        return 0


class _Stream:
    cuda_stream = 17

    def synchronize(self) -> None:
        return None


def _boundary(library: Any) -> AllocatorBoundary:
    return AllocatorBoundary(
        library,
        runtime_handle=0,
        plan_id=1,
        device_ordinal=0,
        telemetry_capacity=1024,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_manifest_hydration_restores_arguments_before_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(failures_module, "runtime_library", _IdleRuntime)
    artifact = _artifact()
    profiler = _profiler(_TaskLibrary())
    profiler.executables.prepare_manifests((artifact,))
    observed_arguments: list[int] = []

    def measure(_profiler: Any, source: Any, **_options: object) -> TaskMeasurement:
        observed_arguments.append(len(source.task.example_arguments))
        return TaskMeasurement(1, 0, 0, (), (1,), "test")

    monkeypatch.setattr(profiler_package, "measure_task", measure)

    profiler.measure(artifact)

    assert observed_arguments == [len(artifact.example_arguments)]
    assert profiler.executables.get(artifact).example_arguments == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_device_measurement_uses_events_and_reports_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(failures_module, "runtime_library", _IdleRuntime)
    profiler = _profiler(_TaskLibrary(), sample_iterations=2)
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
    monkeypatch.setattr(
        measurement_module,
        "measure_workspace",
        lambda boundary, task, stream: WorkspaceObservation(
            workspace,  # type: ignore[arg-type]
            WorkspaceTimings(),
        ),
    )
    monkeypatch.setattr(profiler.boundary, "requested_allocated_bytes", lambda: 0)
    measurement = profiler.measure(_artifact())
    assert measurement.runtime_ns >= 0
    assert len(measurement.samples_ns) == 2
    assert measurement.workspace_charged_bytes == 256
    assert measurement.provenance.startswith("backend-events")
    assert "+torch-inductor" in measurement.provenance

    environment = profile_environment(device_ordinal=0, provider_id="test")
    assert environment.compute_capability == torch.cuda.get_device_capability(0)
    assert environment.provider_id == "test"


def test_workspace_boundary_always_stops_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(failures_module, "runtime_library", _IdleRuntime)
    calls: list[str] = []
    sentinel = object()
    monkeypatch.setattr(
        workspace_module,
        "start_allocation_telemetry",
        lambda library, capacity: calls.append(f"start:{capacity}"),
    )
    monkeypatch.setattr(
        workspace_module,
        "stop_allocation_telemetry",
        lambda library: calls.append("stop"),
    )
    monkeypatch.setattr(
        workspace_module, "read_allocation_telemetry", lambda library: ()
    )
    monkeypatch.setattr(
        workspace_module,
        "summarize_task_workspace",
        lambda events, **options: sentinel,
    )
    boundary = _boundary(_TaskLibrary())
    artifact = _artifact()
    executable = _compiled_task(artifact, lambda *args: torch.ones(1))
    observed = measure_workspace(boundary, executable, _Stream())  # type: ignore[arg-type]
    assert observed.profile is sentinel
    # The capacity is a tuning choice; what this pins is that telemetry starts
    # at the configured size and is always stopped.
    assert calls == [f"start:{boundary.telemetry_capacity}", "stop"]

    failing = _boundary(_TaskLibrary(before_status=5))
    with pytest.raises(CaptureError, match="allocation scope begin"):
        measure_workspace(failing, executable, _Stream())  # type: ignore[arg-type]
    assert calls[-2:] == [f"start:{boundary.telemetry_capacity}", "stop"]


def test_workspace_releases_disposable_results_before_scope_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(failures_module, "runtime_library", _IdleRuntime)
    calls: list[str] = []

    class Result:
        def __del__(self) -> None:
            calls.append("release-result")

    class Library(_TaskLibrary):
        def shadowspill_pytorch_allocation_scope_end(self, *arguments: object) -> int:
            del arguments
            calls.append("end-scope")
            return 0

    monkeypatch.setattr(
        workspace_module, "start_allocation_telemetry", lambda *a, **k: None
    )
    monkeypatch.setattr(
        workspace_module, "stop_allocation_telemetry", lambda *a, **k: None
    )
    monkeypatch.setattr(
        workspace_module, "read_allocation_telemetry", lambda library: ()
    )
    monkeypatch.setattr(
        workspace_module,
        "summarize_task_workspace",
        lambda events, **options: object(),
    )
    artifact = _artifact()
    executable = _compiled_task(artifact, lambda: Result())
    measure_workspace(_boundary(Library()), executable, _Stream())  # type: ignore[arg-type]
    assert calls == ["release-result", "end-scope"]


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

    tensor = torch.empty(4, device="cuda")
    assert output_allocation_views(
        _boundary(_Lookup()), (tensor, tensor.view(2, 2))
    ) == ({91: ((0, 0), (1, 0))}, ())

    class _Missing:
        @staticmethod
        def shadowspill_pytorch_allocation_for_pointer(*arguments: object) -> int:
            del arguments
            return 5

    with pytest.raises(CaptureError, match="outside"):
        output_allocation_views(_boundary(_Missing()), tensor)


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
        TaskProfiler(object(), runtime_handle=0, plan_id=1, device_ordinal=0, **options)


def test_retention_audit_accepts_a_stable_live_byte_baseline() -> None:
    observation = WorkspaceObservation(
        SimpleNamespace(persistent_extent_bytes=(32,)),  # type: ignore[arg-type]
        WorkspaceTimings(execution_ns=5),
    )
    measurements = iter((100, 132, 132, 132))

    audited = audit_workspace_retention(lambda: observation, lambda: next(measurements))

    assert audited.profile is observation.profile
    # Two invocations were needed to see one repeated reading, and the audit
    # charges the caller for both.
    assert audited.timings.execution_ns == 10


def test_retention_audit_rejects_unbounded_growth() -> None:
    observation = WorkspaceObservation(
        SimpleNamespace(persistent_extent_bytes=(32,)),  # type: ignore[arg-type]
        WorkspaceTimings(),
    )
    measurements = iter((100, 132, 164, 196))

    with pytest.raises(AllocationTelemetryError, match="without reaching"):
        audit_workspace_retention(
            lambda: observation,
            lambda: next(measurements),
            maximum_iterations=3,
        )


def test_profiler_rejects_unknown_artifact_protocol() -> None:
    class _Unknown:
        compatibility_digest = "unknown"

    profiler = _profiler()
    artifact = _Unknown()
    with pytest.raises(TypeError, match="unsupported profiling artifact"):
        profiler.measure(artifact)
    with pytest.raises(TypeError, match="unsupported executable artifact"):
        profiler.take_compiled_tasks((artifact,))


def test_compiler_function_transfer_deduplicates_structural_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(failures_module, "runtime_library", _IdleRuntime)
    artifact = _artifact()
    calls: list[str] = []

    def compile_once(value: GraphArtifact, *, device_ordinal: int) -> CompiledTask:
        calls.append(value.compatibility_digest)
        assert device_ordinal == 0
        return _compiled_task(value, lambda *arguments: arguments)

    monkeypatch.setattr(compiler_module, "compile_artifact", compile_once)
    profiler = _profiler(_TaskLibrary())
    indexed = profiler.take_compiled_tasks((artifact, artifact))
    assert tuple(indexed.functions) == (artifact.compatibility_digest,)
    assert tuple(indexed.manifests) == (artifact.compatibility_digest,)
    assert calls == [artifact.compatibility_digest]

    profiler.executables.get(artifact)
    profiler.executables.get(artifact)
    assert calls == [artifact.compatibility_digest, artifact.compatibility_digest]


def test_compiler_failure_has_structural_problem_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact()

    def fail_compile(value: GraphArtifact, *, device_ordinal: int) -> CompiledTask:
        del value, device_ordinal
        raise RuntimeError("compiler exploded")

    monkeypatch.setattr(compiler_module, "compile_artifact", fail_compile)

    with pytest.raises(CompilationError, match="compiler exploded") as captured:
        _profiler().executables.get(artifact)

    assert captured.value.structural_contract == artifact.compatibility_digest
    assert captured.value.task_kind == artifact.kind
    assert captured.value.operators == artifact.operator_targets
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_profile_failure_has_structural_problem_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact()
    monkeypatch.setattr(
        compiler_module,
        "compile_artifact",
        lambda value, *, device_ordinal: _compiled_task(
            value, lambda *items: items, (torch.ones(1),)
        ),
    )

    def explode(*arguments: object, **options: object) -> TaskMeasurement:
        raise RuntimeError("kernel exploded")

    monkeypatch.setattr(profiler_package, "measure_task", explode)

    with pytest.raises(ProfilingError, match="kernel exploded") as captured:
        _profiler().measure(artifact)

    assert captured.value.structural_contract == artifact.compatibility_digest
    assert captured.value.task_kind == artifact.kind
    assert captured.value.operators == artifact.operator_targets
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_measurement_releases_device_examples_between_structural_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact()
    examples = [torch.ones(8)]
    example_reference = weakref.ref(examples[0])

    def compile_with_large_example(
        value: GraphArtifact, *, device_ordinal: int
    ) -> CompiledTask:
        assert device_ordinal == 0
        return _compiled_task(value, lambda *arguments: arguments, (examples[0],))

    measurement = TaskMeasurement(1, 0, 0, (), (1,), "test")
    monkeypatch.setattr(compiler_module, "compile_artifact", compile_with_large_example)
    profiler = _profiler()
    stale_frames: list[object] = []

    def measure_and_retain(
        _profiler: Any, source: Any, **options: object
    ) -> TaskMeasurement:
        del options
        stale_frames.append(source.task)
        return measurement

    monkeypatch.setattr(profiler_package, "measure_task", measure_and_retain)

    observed = profiler.measure(artifact)
    assert observed.runtime_ns == measurement.runtime_ns
    assert observed.workspace_charged_bytes == measurement.workspace_charged_bytes
    assert observed.profiling_wall_time_ns > 0
    examples.clear()
    gc.collect()
    assert example_reference() is None
    assert stale_frames
    assert not stale_frames[0].example_arguments  # type: ignore[attr-defined]
    assert profiler.executables.get(artifact).example_arguments == ()
