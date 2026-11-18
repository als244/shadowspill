"""Saved-plan worker cannot import its model, capture, profile or plan again."""

import sys
from pathlib import Path

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

from engine_tests.execution._cold import forbid_preparation, forbidden

CachingAutotuner.autotune_to_one_config = forbidden


# Executor's shared-plan admission uses static planning helpers. Import these
# before blocking any re-entry to phase preparation during loading/execution.
from shadowspill import Executor, Plan  # noqa: E402

forbid_preparation(
    "shadowspill.planning",
    "engine_tests.execution._replanned_model",
    "shadowspill_workloads.models",
    # A saved Plan is read through its own records: what the search admitted
    # and how they are serialized. Reading them is not planning again.
    allowed=(
        "shadowspill.planning.memory_search",
        "shadowspill.planning.serialization",
    ),
)
directory, gib, kind = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
provider = {
    "embedding": "mlops.providers.builtin.embedding",
    "llama": "mlops.providers.builtin.rope",
}.get(kind)
if provider:
    assert provider not in sys.modules
plan = Plan.load(directory / f"plan-{gib}.ssp")
oracle = torch.load(directory / "oracle.pt", weights_only=True)


def fill(request, chunk):
    assert chunk.nbytes <= 40 and chunk.device.type == "cpu"
    values = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(values[request.offset : request.offset + chunk.numel()])


with Executor(plan) as executor:
    executor.initialize(fill, chunk_bytes=40)
    for loss, expected in oracle["expected"]:
        result = executor.run(
            args=[args for args, _ in oracle["calls"]],
            kwargs=[kwargs for _, kwargs in oracle["calls"]],
        ).result()
        actual_loss = sum(float(value) for value in result.losses)
        del result
        torch.testing.assert_close(actual_loss, loss, rtol=2e-5, atol=2e-6)
        actual = {}

        def collect(request, chunk, actual=actual):
            actual.setdefault(
                request.name,
                torch.empty(
                    request.storage_elements,
                    dtype=request.dtype,
                ),
            )
            actual[request.name][request.offset : request.offset + chunk.numel()].copy_(
                chunk
            )

        executor.export_tensors(collect, chunk_bytes=40)
        for name, value in expected.items():
            torch.testing.assert_close(
                actual[name].reshape(value.shape), value, rtol=2e-4, atol=2e-5
            )
    if provider:
        assert provider in sys.modules
print("Qualified cold saved Executor at", gib, "GiB", flush=True)
