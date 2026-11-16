"""Execute changed packing metadata without capture, compilation or planning."""

import sys
from pathlib import Path

import torch
import triton

from engine_tests.execution._cold import forbid_preparation


def compiled(**event):
    assert event["cache_hit"], f"unexpected compilation: {event['src'].name}"


triton.knobs.compilation.listener = compiled
forbid_preparation()
from shadowspill import Executor, Plan  # noqa: E402
from shadowspill.training.observations import to_host  # noqa: E402

root = Path(sys.argv[1])
oracle = torch.load(root / "oracle.pt", weights_only=True)


def initialize(request, chunk):
    raise AssertionError(f"metadata-only model has no state to load: {request.name}")


with Executor(Plan.load(root / "plan.ssp")) as executor:
    executor.initialize(initialize)
    for arguments, expected in zip(oracle["calls"], oracle["expected"], strict=True):
        actual = executor.run(args=(arguments,)).result()
        torch.testing.assert_close(to_host(actual), expected, rtol=0, atol=0)
        del actual
    executor.synchronize()
print("Verified four packings with one saved Plan", flush=True)
