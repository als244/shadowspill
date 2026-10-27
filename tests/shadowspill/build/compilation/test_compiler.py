"""Packaged code retains its optimized contract and survives compiler cache reuse."""

import torch
from torch import nn

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
