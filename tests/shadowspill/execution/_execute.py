"""Worker intentionally has no model and cannot build, measure or calibrate."""

import importlib.abc
import sys
from pathlib import Path

import torch
import torch._inductor.compile_fx as compiler
import triton
from torch.utils._pytree import tree_map

from shadowspill import Plan
from shadowspill.runtime import Runtime


def forbidden(*args, **kwargs):
    raise AssertionError("execution tried to compile or calibrate")


class BlockPhases(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == name or fullname.startswith(name + ".")
            for name in (
                "shadowspill.build",
                "shadowspill.store",
                "shadowspill.profiling.measure",
                "shadowspill.pytorch.capture",
                "shadowspill.pytorch.compilation",
            )
        ):
            raise AssertionError(f"execution imported {fullname}")


compiler.compile_fx = compiler.compile_fx_inner = triton.compile = (
    Runtime._calibrate
) = forbidden
sys.meta_path.insert(0, BlockPhases())
from shadowspill.execution import Executor  # noqa: E402

directory = Path(sys.argv[1])
plan = Plan.load(directory / "plan.ssp")
oracle = torch.load(directory / "oracle.pt", weights_only=True)


def fill(request, chunk):
    assert chunk.device.type == "cpu" and chunk.numel() * chunk.element_size() <= 40
    source = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(source[request.offset : request.offset + chunk.numel()])


def cpu(value):
    return tree_map(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, value)


executor = Executor(plan)
executor.initialize(fill, chunk_bytes=40)
previous, previous_value = None, None
for outputs, state in oracle["expected"]:
    pending = executor.run(input_loader=lambda i: oracle["examples"][i])
    result = pending.result()
    actual_outputs = (
        result
        if plan.program.bindings["kind"] == "forward"
        else list(zip(result.losses, result.metrics, strict=True))
    )
    torch.testing.assert_close(cpu(actual_outputs), outputs)
    if previous is not None:
        torch.testing.assert_close(cpu(previous), previous_value)
    previous, previous_value = actual_outputs, cpu(actual_outputs)
    actual = {}

    def collect(request, chunk, actual=actual):
        if request.name not in actual:
            actual[request.name] = torch.empty(
                request.storage_elements, dtype=request.dtype
            )
        actual[request.name][request.offset : request.offset + chunk.numel()].copy_(
            chunk
        )

    executor.export_tensors(collect, chunk_bytes=40)
    assert actual.keys() == state.keys()
    for name, expected in state.items():
        torch.testing.assert_close(actual[name].reshape(expected.shape), expected)
del pending, result, actual_outputs, previous
executor.close()
print("Qualified three saved-plan invocations", flush=True)
