"""Packaged code retains its optimized contract and survives compiler cache reuse."""

import pytest
import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.build.compilation import compile_task
from shadowspill.build.compilation.inductor import compiler
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.executable import restore_executable


class Add(nn.Module):
    def forward(self, left, right):
        return left + right


class MultiplyByOne(nn.Module):
    def forward(self, value):
        return value * torch.ones_like(value)


def artifact(model, inputs):
    return GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(model),
        example_inputs=inputs,
    )


def test_warm_aot_cache_retains_code_geometry_and_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    inputs = (torch.randn(8, 8), torch.randn(8, 8))
    captured = artifact(Add(), inputs)
    first = compile_task(captured)

    def unexpected(*args, **kwargs):
        raise AssertionError("warm AOT cache rebuilt GraphLowering")

    monkeypatch.setattr(compiler, "_graph_lowering_contract", unexpected)
    second = compile_task(captured)
    for built in (first, second):
        (output,) = restore_executable(built.payload)(*inputs)
        torch.testing.assert_close(output, inputs[0] + inputs[1])
    assert first.manifest == second.manifest
    assert first.memory == second.memory
    assert not (tmp_path / "shadowspill" / "task_manifests").exists()


def test_manifest_describes_optimized_output_alias():
    value = torch.randn(32)
    captured = artifact(MultiplyByOne(), (value,))
    assert captured.storage_contract.roots[0].kind.value == "fresh"
    built = compile_task(captured)
    root = built.manifest.storage_contract.roots[0]
    assert root.kind.value == "input" and root.source_input == 0
    (output,) = restore_executable(built.payload)(value)
    assert output.data_ptr() == value.data_ptr()
    torch.testing.assert_close(output, value)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_input_alias_outputs_use_storage_origin_before_and_after_input(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    def operation(later, earlier):
        return later.t(), earlier.t(), later.sum()

    storage = torch.arange(24, dtype=torch.float32, device=device).reshape(6, 4)
    inputs = (storage[3:], storage[:3])
    captured = GraphArtifact.capture(
        kind="inference",
        graph_module=make_fx(operation)(*inputs),
        example_inputs=inputs,
    )
    built = compile_task(captured)
    contract = built.manifest.storage_contract
    later, earlier, fresh = contract.output_views
    assert later.root_id == earlier.root_id != fresh.root_id
    root = contract.roots[later.root_id]
    assert root.kind.value == "input" and root.source_input == 0
    assert later.offset_bytes == inputs[0].storage_offset() * storage.element_size()
    assert earlier.offset_bytes == 0
    assert root.minimum_span_bytes == storage.untyped_storage().nbytes()
    assert built.manifest.root_allocations[later.root_id].requested_bytes == 0

    executable = restore_executable(built.payload)
    for values in (storage, storage + 100):
        actual = executable(values[3:], values[:3])
        torch.testing.assert_close(actual, operation(values[3:], values[:3]))
        for output, view in zip(actual[:2], (later, earlier), strict=True):
            assert (
                output.untyped_storage().data_ptr()
                == values.untyped_storage().data_ptr()
            )
            assert output.storage_offset() * output.element_size() == view.offset_bytes
