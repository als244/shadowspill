from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_forward
from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    TaskInputProvenance,
    TaskInputRole,
    capture_forward_stage_artifacts,
)
from shadowspill.pytorch.capture.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.partition import partition_export
from shadowspill.pytorch.profiling import (
    ProfileEnvironment,
    ProfileKey,
    ProfileRepository,
    TaskAllocationContract,
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
    profile_unique_artifacts,
)


def test_allocation_probe_matrix_is_part_of_profile_identity() -> None:
    environment = _environment()
    first = ProfileKey(
        "a" * 64,
        environment,
        allocation_probe_seeds=1,
        allocation_probe_repetitions=2,
    )
    more_seeds = ProfileKey(
        "a" * 64,
        environment,
        allocation_probe_seeds=3,
        allocation_probe_repetitions=2,
    )
    more_repetitions = ProfileKey(
        "a" * 64,
        environment,
        allocation_probe_seeds=1,
        allocation_probe_repetitions=4,
    )

    assert len({first.digest, more_seeds.digest, more_repetitions.digest}) == 3


def test_task_measurement_rejects_legacy_allocation_schema() -> None:
    measurement = TaskMeasurement(1, 0, 0, (), (1,), "test")
    legacy = measurement.to_dict()
    legacy["allocation_abi"] = legacy.pop("allocation_contract")

    with pytest.raises(ValueError, match="invalid schema"):
        TaskMeasurement.from_dict(legacy)


class _Repeated(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(6)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = torch.relu(layer(value))
        return value


def _artifacts() -> tuple[GraphArtifact, ...]:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(_Repeated(), mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        partitioned = partition_export(capture_forward(model, inputs), model)
        return capture_forward_stage_artifacts(partitioned)


def _environment() -> ProfileEnvironment:
    return ProfileEnvironment(
        torch_version="2.13.0",
        cuda_version="13.0",
        device_name="test-device",
        compute_capability=(12, 0),
        compiler_id="inductor",
        provider_id="aten",
    )


def test_structural_profile_runs_once_and_warm_cache_runs_nothing(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    calls: list[str] = []

    def measure(artifact: GraphArtifact) -> TaskMeasurement:
        calls.append(artifact.compatibility_digest)
        return TaskMeasurement(
            runtime_ns=100,
            workspace_requested_bytes=64,
            workspace_charged_bytes=64,
            workspace_extent_bytes=(64,),
            samples_ns=(90, 100, 110),
            provenance="unit-test",
            allocation_trace=(
                TaskAllocationEvent(
                    0,
                    TaskAllocationOperation.ALLOCATE,
                    64,
                    64,
                ),
                TaskAllocationEvent(
                    0,
                    TaskAllocationOperation.FREE,
                    64,
                    64,
                ),
            ),
            persistent_extent_bytes=(32,),
        )

    cache = ProfileRepository(tmp_path)
    cold = profile_unique_artifacts(
        artifacts,
        environment=_environment(),
        measure=measure,
        cache=cache,
    )
    # Semantic roles do not split an otherwise identical physical profile.
    # Authentic non-floating contents still do.
    assert cold.unique_keys == 1
    assert cold.cache_hits == 0
    assert cold.cache_misses == 1
    assert len(calls) == 1
    assert len(cold.measurements) == 6
    assert cold.fixed_slab_bytes == 32

    warm = profile_unique_artifacts(
        artifacts,
        environment=_environment(),
        measure=lambda artifact: (_ for _ in ()).throw(AssertionError(artifact)),
        cache=cache,
    )
    assert warm.cache_hits == 1
    assert warm.cache_misses == 0
    assert warm.measurements == cold.measurements
    assert warm.fixed_slab_bytes == 32


def test_profile_environment_changes_cache_identity(tmp_path: Path) -> None:
    artifacts = _artifacts()[:1]
    calls = 0

    def measure(artifact: GraphArtifact) -> TaskMeasurement:
        nonlocal calls
        calls += 1
        return TaskMeasurement(1, 0, 0, (), (1,), "test")

    cache = ProfileRepository(tmp_path)
    profile_unique_artifacts(
        artifacts,
        environment=_environment(),
        measure=measure,
        cache=cache,
    )
    changed = ProfileEnvironment(
        torch_version="2.13.0",
        cuda_version="13.0",
        device_name="test-device",
        compute_capability=(12, 0),
        compiler_id="inductor",
        provider_id="custom",
    )
    profile_unique_artifacts(
        artifacts,
        environment=changed,
        measure=measure,
        cache=cache,
    )
    assert calls == 2


def test_profile_identity_ignores_control_contents(
    tmp_path: Path,
) -> None:
    """One profile per (indexed contract, declared metadata).

    Input values are never inspected: artifacts sharing a contract and
    metadata share one representative measurement even when their
    integer control contents differ. Structure the caller wants
    distinguished must be declared through profiling metadata (covered
    by test_profiling_metadata_splits_measurements_without_recompiling
    _identity).
    """
    module = torch.fx.symbolic_trace(nn.Identity())

    def artifact(value: torch.Tensor) -> GraphArtifact:
        return GraphArtifact.capture(
            kind="inference",
            graph_module=module,
            example_inputs=(value,),
            input_provenance=(
                TaskInputProvenance(
                    TaskInputRole.CONTROL,
                    "metadata",
                    representative_value=value,
                ),
            ),
        )

    first_value = torch.tensor([0, 13, 32, 64], dtype=torch.int64)
    second_value = torch.tensor([0, 17, 48, 96], dtype=torch.int64)
    artifacts = (
        artifact(first_value),
        artifact(first_value.clone()),
        artifact(second_value),
    )
    calls = 0

    def measure(candidate: GraphArtifact) -> TaskMeasurement:
        nonlocal calls
        calls += 1
        trace = ()
        return TaskMeasurement(
            calls,
            24 if trace else 0,
            256 if trace else 0,
            (256,) if trace else (),
            (calls,),
            "problem-test",
            allocation_trace=trace,
            allocation_contract=TaskAllocationContract.capture(trace),
        )

    result = profile_unique_artifacts(
        artifacts,
        environment=_environment(),
        measure=measure,
        cache=ProfileRepository(tmp_path),
    )
    assert calls == 1
    assert result.unique_keys == 1
    assert len({item.compatibility_digest for item in artifacts}) == 1
    assert result.measurements[0] is result.measurements[1]
    assert result.measurements[0] is result.measurements[2]


def test_profile_identity_accepts_scalar_integer_control(tmp_path: Path) -> None:
    module = torch.fx.symbolic_trace(nn.Identity())
    value = torch.tensor(1, dtype=torch.int64)
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=module,
        example_inputs=(value,),
        input_provenance=(
            TaskInputProvenance(
                TaskInputRole.CONTROL,
                "step",
                representative_value=value,
            ),
        ),
    )
    result = profile_unique_artifacts(
        (artifact,),
        environment=_environment(),
        measure=lambda _artifact: TaskMeasurement(1, 0, 0, (), (1,), "scalar"),
        cache=ProfileRepository(tmp_path),
    )
    assert result.unique_keys == 1


def test_invalid_cached_physical_profile_is_remeasured(tmp_path: Path) -> None:
    artifact = _artifacts()[0]
    cache = ProfileRepository(tmp_path)
    calls = 0

    def measure(_artifact: GraphArtifact) -> TaskMeasurement:
        nonlocal calls
        calls += 1
        return TaskMeasurement(2, 0, 0, (), (2,), "fresh")

    profile_unique_artifacts(
        (artifact,),
        environment=_environment(),
        measure=lambda _artifact: TaskMeasurement(1, 0, 0, (), (1,), "stale"),
        cache=cache,
    )

    def validate(
        _artifact: GraphArtifact,
        measurement: TaskMeasurement,
    ) -> None:
        if measurement.runtime_ns != 2:
            raise CaptureError("stale physical output extent")

    result = profile_unique_artifacts(
        (artifact,),
        environment=_environment(),
        measure=measure,
        cache=cache,
        validate=validate,
    )
    assert calls == 1
    assert result.cache_hits == 0
    assert result.cache_misses == 1
    assert result.measurements[0].runtime_ns == 2


def test_profiling_metadata_splits_measurements_without_recompiling_identity(
    tmp_path: Path,
) -> None:
    artifact = _artifacts()[0]
    calls = 0

    def measure(_artifact: GraphArtifact) -> TaskMeasurement:
        nonlocal calls
        calls += 1
        return TaskMeasurement(
            calls,
            0,
            0,
            (),
            (calls,),
            "metadata-test",
            persistent_extent_bytes=(32,),
        )

    cache = ProfileRepository(tmp_path)
    cold = profile_unique_artifacts(
        (artifact, artifact),
        environment=_environment(),
        measure=measure,
        cache=cache,
        profiling_metadata_digests=("a" * 64, "b" * 64),
    )
    assert calls == 2
    assert cold.unique_keys == 2
    assert len(set(cold.key_digests)) == 2
    assert cold.fixed_slab_bytes == 32

    warm = profile_unique_artifacts(
        (artifact, artifact),
        environment=_environment(),
        measure=lambda item: (_ for _ in ()).throw(AssertionError(item)),
        cache=cache,
        profiling_metadata_digests=("a" * 64, "b" * 64),
    )
    assert warm.cache_hits == 2
    assert warm.cache_misses == 0
