"""Worker intentionally has no model and cannot build, measure or calibrate."""

import importlib.abc
import sys
from pathlib import Path

import pytest
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


compiler.compile_fx = compiler.compile_fx_inner = Runtime._calibrate = forbidden
sys.meta_path.insert(0, BlockPhases())
from qualification.runtime_evidence import adapter_statistics  # noqa: E402
from shadowspill import Executor, load_checkpoint, save_checkpoint  # noqa: E402
from tests.shadowspill.execution._stream_checks import StreamChecks  # noqa: E402

directory = Path(sys.argv[1])
plan = Plan.load(directory / "plan.ssp")
oracle = torch.load(directory / "oracle.pt", weights_only=True)
lazy_kernels = set(oracle.get("lazy_kernels", ()))
norms = oracle.get("parameter_metrics", ())
norm_kernels = {"norm_partials", "norm_sum"} if norms else set()
compiled_kernels = []


def compilation_finished(**event):
    # Patching triton.compile misses the compiler imported inside JITFunction.
    # This listener sees real compilation, including that internal entry point.
    if not event["cache_hit"]:
        name = event["src"].name
        assert name in lazy_kernels | norm_kernels, (
            f"execution compiled unexpected kernel: {name}"
        )
        compiled_kernels.append(name)


triton.knobs.compilation.listener = compilation_finished


def fill(request, chunk):
    assert chunk.device.type == "cpu" and chunk.numel() * chunk.element_size() <= 40
    source = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(source[request.offset : request.offset + chunk.numel()])


def cpu(value):
    return tree_map(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, value)


executor = Executor(plan, parameter_norms=bool(norms))
prepared_norm_kernels = [name for name in compiled_kernels if name in norm_kernels]
statistics = adapter_statistics()
assert statistics.physical_budget_sealed
assert statistics.runtime.event_lease_sealed
streams = StreamChecks(executor) if oracle.get("check_task_streams") else None
executor.initialize(fill, chunk_bytes=40, restore=oracle["restore"])
assert (
    sum(executor.runtime._installed.admitted_layout_bytes.values())
    == plan.memory_layout.fixed_slice_bytes
)
with pytest.raises(ValueError, match="unknown runtime hyperparameters"):
    executor.run(
        input_loader=lambda i: oracle["examples"][i], hyperparams={"unknown": 1}
    )
previous, previous_value = None, None
previous_norms, previous_norm_values = None, None
first_invocation_compiles = None
steady_statistics = None
for index, (outputs, state) in enumerate(oracle["expected"]):
    with torch.cuda.stream(None if streams is None else streams.begin(index)):
        pending = executor.run(
            input_loader=lambda i: oracle["examples"][i],
            hyperparams=oracle["hyperparams"][index],
            accumulate_gradients=oracle["accumulate"][index],
            collect_parameter_norms=bool(norms and norms[index] is not None),
        )
        if streams is not None:
            streams.check()
    result = pending.result()
    assert [
        name for name in compiled_kernels if name in norm_kernels
    ] == prepared_norm_kernels
    if norms:
        expected_norms = norms[index] or {}
        torch.testing.assert_close(
            result.parameter_metrics, expected_norms, rtol=2e-6, atol=1e-6
        )
        if previous_norms is not None:
            torch.testing.assert_close(
                previous_norms, previous_norm_values, rtol=0, atol=0
            )
        previous_norms = result.parameter_metrics
        previous_norm_values = tree_map(lambda t: t.clone(), previous_norms)
    if first_invocation_compiles is None:
        first_invocation_compiles = len(compiled_kernels)
    else:
        assert len(compiled_kernels) == first_invocation_compiles, compiled_kernels
    actual_outputs = (
        result
        if plan.program.bindings["kind"] == "forward"
        else list(zip(result.losses, result.metrics, strict=True))
    )
    torch.testing.assert_close(cpu(actual_outputs), outputs)
    if previous is not None:
        torch.testing.assert_close(cpu(previous), previous_value)
    if isinstance(result, dict) and "view" in result:
        assert (
            result["input"].untyped_storage()._cdata
            == result["view"].untyped_storage()._cdata
        )
        result["weight"].zero_()  # A caller snapshot must not change stored parameters.
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
    statistics = adapter_statistics()
    if steady_statistics is not None:
        assert (
            statistics.runtime.event_lease_driver_creates
            == steady_statistics.runtime.event_lease_driver_creates
        )
        assert (
            statistics.backend.device_allocations
            == steady_statistics.backend.device_allocations
        )
    assert actual.keys() == state.keys(), {
        "unexpected": actual.keys() - state.keys(),
        "missing": state.keys() - actual.keys(),
    }
    for name, expected in state.items():
        torch.testing.assert_close(actual[name].reshape(expected.shape), expected)
    if index == 0:
        checkpoint = directory / "checkpoint.ssp"
        save_checkpoint(executor, checkpoint, metadata={"completed": 1}, chunk_bytes=40)
        executor.initialize(lambda request, chunk: chunk.zero_(), restore=True)
        assert load_checkpoint(executor, checkpoint, chunk_bytes=28) == {"completed": 1}
        steady_statistics = adapter_statistics()
if plan.program.bindings["kind"] == "forward":
    first = executor.run(
        args=[oracle["examples"][0][0]], kwargs=[oracle["examples"][0][1]]
    )
    assert not first.resolved
    second = executor.run(input_loader=lambda i: oracle["examples"][i])
    assert first.resolved
    torch.testing.assert_close(cpu(second.result()), oracle["expected"][-1][0])
    del first, second
with pytest.raises(Exception, match="caller-owned device outputs"):
    executor.close()
del pending, result, actual_outputs, previous
executor.close()
assert set(compiled_kernels) - norm_kernels == lazy_kernels, compiled_kernels
print("Qualified three saved-plan invocations", flush=True)
