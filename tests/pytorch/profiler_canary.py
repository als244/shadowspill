"""Fresh-process compiled task profiling through the production allocator."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch._allocator import install_allocator
from shadowspill.pytorch.aot import capture_forward
from shadowspill.pytorch.capture import capture_forward_stage_artifacts
from shadowspill.pytorch.compiler import CudaTaskProfiler, profile_environment
from shadowspill.pytorch.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.materialization import flat_runtime_arguments
from shadowspill.pytorch.partition import partition_export
from shadowspill.pytorch.profiling import ProfileCache, profile_unique_artifacts


class _Repeated(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(512, 512, bias=False) for _ in range(2)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = torch.relu(layer(value))
        return value


def main() -> int:
    adapter_path = Path(sys.argv[1]).resolve()
    installed = install_allocator(
        adapter_path,
        device_ordinal=0,
        device_budget_bytes=2 << 30,
        provider_headroom_bytes=512 << 20,
        spill_pool_bytes=64 << 20,
    )
    real_model = _Repeated()
    real_inputs = (torch.randn(16, 512),)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(real_model, mode)
    inputs = fake_cuda_inputs(real_inputs, mode)
    with mode, torch.no_grad():
        capture = capture_forward(model, inputs)
        representative_roots = tuple(
            value.detach() if isinstance(value, torch.Tensor) else value
            for value in flat_runtime_arguments(capture, real_model, real_inputs)
        )
        artifacts = capture_forward_stage_artifacts(
            partition_export(
                capture,
                model,
                representative_root_inputs=representative_roots,
            )
        )
    if len(artifacts) != 2:
        raise AssertionError("canary did not produce two task positions")

    profiler = CudaTaskProfiler(
        installed.library,
        device_ordinal=0,
        warmup_iterations=2,
        sample_iterations=3,
    )
    with tempfile.TemporaryDirectory() as directory:
        cache = ProfileCache(directory)
        measured = profile_unique_artifacts(
            artifacts,
            environment=profile_environment(
                device_ordinal=0, provider_id="pytorch-aten"
            ),
            measure=profiler.measure,
            cache=cache,
        )
        if measured.unique_keys != 1 or measured.cache_misses != 1:
            raise AssertionError(
                "structurally equal tasks were profiled more than once"
            )
        if measured.measurements[0].runtime_ns <= 0:
            raise AssertionError("CUDA event timing was empty")
        # These artifacts are already explicit no-grad tasks. A simple fused
        # linear/relu stage may have zero anonymous workspace; compiling it as
        # a second autograd graph used to create hidden saved-output storage
        # and made this assertion pass for the wrong reason. Require the
        # returned allocation timeline instead.
        if not measured.measurements[0].allocation_trace:
            raise AssertionError("task allocation telemetry saw no allocations")
        warm = profile_unique_artifacts(
            artifacts,
            environment=profile_environment(
                device_ordinal=0, provider_id="pytorch-aten"
            ),
            measure=lambda artifact: (_ for _ in ()).throw(AssertionError(artifact)),
            cache=cache,
        )
        if warm.cache_hits != 1 or warm.cache_misses != 0:
            raise AssertionError("warm profiling did not use the content cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
