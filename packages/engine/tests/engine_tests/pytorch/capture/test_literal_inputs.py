"""Exported constants retain their original values and device placement."""

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export.graph_signature import InputKind

from shadowspill.pytorch.capture.export import capture_forward
from shadowspill.pytorch.capture.inputs import flat_runtime_arguments


class Literal(nn.Module):
    def forward(self, x):
        bounds = torch.tensor([0, x.shape[0]], device="cpu")
        return x + bounds.sum()


class Attribute(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.tensor([2.0, 3.0])

    def forward(self, x):
        return x + self.offset


def test_inline_cpu_literal_is_read_from_export():
    model, x = Literal(), torch.zeros(3, 2)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        captured = capture_forward(model, (mode.from_tensor(x),))
    args = flat_runtime_arguments(captured, model, (x,))
    constants = [
        value
        for spec, value in zip(
            captured.exported_program.graph_signature.input_specs, args, strict=True
        )
        if spec.kind is InputKind.CONSTANT_TENSOR
    ]
    assert len(constants) == 1
    torch.testing.assert_close(constants[0], torch.tensor([0, 3]))
    torch.testing.assert_close(
        captured.exported_program.graph_module(*args)[0], model(x)
    )


def test_attribute_constant_still_uses_original_values():
    model, x = Attribute(), torch.zeros(3, 2)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        captured = capture_forward(model, (mode.from_tensor(x),))
    args = flat_runtime_arguments(captured, model, (x,))
    torch.testing.assert_close(
        captured.exported_program.graph_module(*args)[0], model(x)
    )
