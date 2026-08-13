from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.aot import capture_forward
from shadowspill.pytorch.capture import GraphArtifact
from shadowspill.pytorch.contracts import CaptureError
from shadowspill.pytorch.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.partition import capture_forward_stages, partition_export
from shadowspill.pytorch.profiling import (
    ProfileCache,
    ProfileEnvironment,
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
    profile_unique_artifacts,
)


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
        return capture_forward_stages(partitioned)


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

    cache = ProfileCache(tmp_path)
    cold = profile_unique_artifacts(
        artifacts,
        environment=_environment(),
        measure=measure,
        cache=cache,
    )
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

    cache = ProfileCache(tmp_path)
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


def test_invalid_cached_physical_profile_is_remeasured(tmp_path: Path) -> None:
    artifact = _artifacts()[0]
    cache = ProfileCache(tmp_path)
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
