"""Inline constants remain usable through a caller-selected forward."""

import io

import pytest
import torch
from torch import nn
from torch.export.graph_signature import InputKind

from shadowspill.pytorch.capture.aot import capture_forward
from shadowspill.pytorch.planning.forward.capture import _select_forward


class _WithConstants(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, -0.25]), requires_grad=False)
        self.register_buffer("_shadowspill_constant_0", torch.tensor([0.25, 1.0]))
        self.frequencies = (1.0, 0.125)

    def forward(self, x):
        constant = torch.tensor(self.frequencies, device=x.device)
        return x * self.weight + constant + self._shadowspill_constant_0


@pytest.mark.parametrize("callback", (False, True))
def test_inline_constants_capture_decompose_and_archive(callback):
    model = _WithConstants()
    x = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    expected = model(x)
    if callback:
        _select_forward(model, lambda module, value: module(value))
    captured = capture_forward(model, [x])
    program = captured.exported_program
    state_targets = {
        spec.target
        for spec in program.graph_signature.input_specs
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
    }
    assert state_targets == {"weight", "_shadowspill_constant_0"}
    assert all(
        all(part.isidentifier() for part in name.split("."))
        for name in program.constants
    )
    torch.testing.assert_close(program.module()(x), expected, rtol=0, atol=0)
    stream = io.BytesIO()
    torch.export.save(program, stream)
    stream.seek(0)
    restored = torch.export.load(stream)
    torch.testing.assert_close(restored.module()(x), expected, rtol=0, atol=0)


def test_registered_sequence_constant_paths_are_preserved():
    from shadowspill.pytorch.capture.aot import _repair_constant_targets

    class Offset(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.tensor([0.25, -0.75])

        def forward(self, x):
            return x + self.offset

    model = nn.Sequential(Offset())
    x = torch.ones(2)
    program = torch.export.export(model, (x,), strict=True)
    original = set(program.constants)
    assert any("0." in name for name in original)
    _repair_constant_targets(program)
    assert set(program.constants) == original
    torch.testing.assert_close(program.run_decompositions().module()(x), model(x))
