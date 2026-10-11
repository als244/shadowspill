"""Logical tensor functionalization preserves nested Python control inputs."""

import copy

import pytest
import torch
from torch import nn

from shadowspill.pytorch.capture.aot import capture_forward
from tests.shadowspill.pytorch.state.representations import model


@torch.library.custom_op(
    "test_functional_controls::with_info",
    mutates_args=(),
    schema="(Tensor value) -> (Tensor, Tensor?, SymInt)",
)
def with_info(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    return value.clone(), None, value.shape[0]


@with_info.register_fake
def _with_info_fake(value):
    return torch.empty_like(value), None, value.shape[0]


class Network(nn.Module):
    def __init__(self, *, include_info=False):
        super().__init__()
        self.projection = model()
        self.include_info = include_info
        self.register_buffer("total", torch.zeros(()))

    def forward(self, x, controls):
        self.total.add_(x.sum())
        start, stop, scale, enabled, _label, _unused = controls
        value = self.projection(x)[start:stop]
        if self.include_info:
            value, _unused_tensor, _unused_size = with_info(value)
        return value * scale if enabled else value


@pytest.mark.parametrize("strict", (True, False))
@pytest.mark.parametrize("include_info", (False, True))
def test_wrapper_and_mutation_keep_nested_static_controls(
    monkeypatch, strict, include_info
):
    torch._dynamo.reset()
    original_export = torch.export.export

    def selected_export(*args, **kwargs):
        kwargs["strict"] = strict
        return original_export(*args, **kwargs)

    monkeypatch.setattr(torch.export, "export", selected_export)
    net = Network(include_info=include_info)
    expected = copy.deepcopy(net)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    controls = (0, 2, 0.5, True, "window", None)
    capture = capture_forward(net, [x, controls])
    program = capture.exported_program
    program.validate()
    placeholders = [n for n in program.graph.nodes if n.op == "placeholder"]
    static = [
        n.meta["val"]
        for n in placeholders
        if not isinstance(n.meta["val"], torch.Tensor)
    ]
    assert static == list(controls)
    exported = program.module()
    for value in (x, x * 2):
        torch.testing.assert_close(exported(value, controls), expected(value, controls))
        torch.testing.assert_close(exported.total, expected.total)
    torch._dynamo.reset()
