"""Saved-plan worker cannot import its model, capture, profile or plan again."""

import importlib.abc
import sys
from pathlib import Path

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner


def forbidden(*args, **kwargs):
    raise AssertionError("saved execution attempted Inductor tuning")


CachingAutotuner.autotune_to_one_config = forbidden


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == key or fullname.startswith(key + ".")
            for key in (
                "shadowspill.build",
                "shadowspill.profiling.measure",
                "shadowspill.planning",
                "shadowspill.training._legacy_trainer",
                "shadowspill.store",
                "tests.shadowspill.training._replanned_model",
                "workloads.models",
            )
        ):
            raise AssertionError("execution imported " + fullname)


# Executor's shared-plan admission uses static planning helpers. Import these
# before blocking any re-entry to phase preparation during loading/execution.
from shadowspill import Plan, Trainer  # noqa: E402

sys.meta_path.insert(0, Block())
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


with Trainer(plan) as trainer:
    trainer.initialize(fill, chunk_bytes=40)
    for loss, expected in oracle["expected"]:
        result = trainer.step(
            args=[args for args, _ in oracle["calls"]],
            kwargs=[kwargs for _, kwargs in oracle["calls"]],
        )
        torch.testing.assert_close(result.loss, loss, rtol=2e-5, atol=2e-6)
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

        trainer.executor.export_tensors(collect, chunk_bytes=40)
        for name, value in expected.items():
            torch.testing.assert_close(
                actual[name].reshape(value.shape), value, rtol=2e-4, atol=2e-5
            )
    assert trainer.step_count == 2
    if provider:
        assert provider in sys.modules
print("Qualified cold saved Trainer at", gib, "GiB", flush=True)
