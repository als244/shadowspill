"""Worker intentionally has no model and cannot build, measure or calibrate."""

import ctypes
import sys
from pathlib import Path

import pytest
import torch
import triton
from torch.utils._pytree import tree_map

from shadowspill import Plan
from tests.shadowspill.execution._cold import forbid_preparation

forbid_preparation(
    "tests.shadowspill.execution.optimizers",
    "tests.shadowspill.execution.test_model_controls",
)
from qualification.runtime_evidence import adapter_statistics  # noqa: E402
from shadowspill import Executor, load_checkpoint, save_checkpoint  # noqa: E402
from tests.shadowspill.execution._stream_checks import StreamChecks  # noqa: E402
from tests.shadowspill.execution._trace_checks import check_trace  # noqa: E402
from tests.spill_pool import spill_pool  # noqa: E402

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


def assert_state(executor, state):
    actual = {}

    def collect(request, chunk):
        assert chunk.device.type == "cpu" and chunk.nbytes <= 40
        if request.name not in actual:
            actual[request.name] = torch.empty(
                request.storage_elements, dtype=request.dtype
            )
        actual[request.name][request.offset : request.offset + chunk.numel()].copy_(
            chunk
        )

    executor.export_tensors(collect, chunk_bytes=40)
    assert actual.keys() == state.keys(), {
        "unexpected": actual.keys() - state.keys(),
        "missing": state.keys() - actual.keys(),
    }
    for name, expected in state.items():
        torch.testing.assert_close(
            actual[name].reshape(expected.shape),
            expected,
            msg=lambda error, name=name: f"{name}: {error}",
        )


executor = Executor(
    plan, parameter_norms=bool(norms), spill=spill_pool(plan.memory_budgets.spill_bytes)
)
annotations_enabled = executor._paths[
    0
].bridge.runtime_library.shadowspill_profiler_annotations_enabled
annotations_enabled.argtypes = [ctypes.c_void_p]
annotations_enabled.restype = ctypes.c_uint8
assert not annotations_enabled(executor.runtime._runtime_handle)
prepared_norm_kernels = [name for name in compiled_kernels if name in norm_kernels]
statistics = adapter_statistics()
assert statistics.physical_budget_sealed
assert statistics.runtime.event_lease_sealed
streams = StreamChecks(executor) if oracle.get("check_task_streams") else None
executor.initialize(fill, chunk_bytes=40, restore=oracle["restore"])
if "initial_state" in oracle:
    assert_state(executor, oracle["initial_state"])
assert (
    sum(executor.runtime._installed.admitted_layout_bytes.values())
    == plan.memory_layout.fixed_slice_bytes
)
with pytest.raises(ValueError, match="unknown runtime hyperparameters"):
    executor.run(
        input_loader=lambda i: oracle["examples"][i], hyperparams={"unknown": 1}
    )
for values in oracle.get("invalid_hyperparams", ()):
    with pytest.raises(ValueError, match="unknown runtime hyperparameters"):
        executor.run(input_loader=lambda i: oracle["examples"][i], hyperparams=values)
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
            profiler_annotations=index == 1,
            runtime_trace=index == 0,
        )
        if streams is not None:
            streams.check()
    assert bool(annotations_enabled(executor.runtime._runtime_handle)) == (index == 1)
    result = pending.result()
    if index == 0:
        check_trace(plan, pending.diagnostics)
    else:
        assert pending.diagnostics is None
    assert not annotations_enabled(executor.runtime._runtime_handle)
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
    if oracle.get("host_metrics"):
        assert all(metric["bounds"].device.type == "cpu" for metric in result.metrics)
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
    assert_state(executor, state)
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
    if index == 0:
        checkpoint = directory / "checkpoint.ssp"
        before_checkpoint = adapter_statistics()
        save_checkpoint(executor, checkpoint, metadata={"completed": 1}, chunk_bytes=40)
        executor.initialize(lambda request, chunk: chunk.zero_(), restore=True)
        assert load_checkpoint(executor, checkpoint, chunk_bytes=28) == {"completed": 1}
        steady_statistics = adapter_statistics()
        assert (
            steady_statistics.allocation_callbacks
            == before_checkpoint.allocation_callbacks
        )
        assert steady_statistics.free_callbacks == before_checkpoint.free_callbacks
if plan.program.bindings["kind"] == "forward":
    first = executor.run(
        args=[oracle["examples"][0][0]], kwargs=[oracle["examples"][0][1]]
    )
    assert not first.resolved
    second = executor.run(input_loader=lambda i: oracle["examples"][i])
    assert first.resolved
    queued = oracle.get("queued_outputs", [oracle["expected"][-1][0]] * 2)
    torch.testing.assert_close(cpu(first.result()), queued[0])
    torch.testing.assert_close(cpu(second.result()), queued[1])
    del first, second
with pytest.raises(Exception, match="caller-owned device outputs"):
    executor.close()
del pending, result, actual_outputs, previous
executor.close()
assert set(compiled_kernels) - norm_kernels == lazy_kernels, compiled_kernels
print("Qualified three saved-plan invocations", flush=True)
