"""Static leaves keep their positions without acquiring tensor storage."""

from __future__ import annotations

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export.graph_signature import InputKind

from shadowspill.pytorch.capture.aot import capture_forward
from shadowspill.pytorch.capture.artifacts import capture_forward_stage_artifacts
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.lowering.forward import lower_partitioned_forward_program
from shadowspill.pytorch.materialization.forward import flat_runtime_arguments
from shadowspill.pytorch.partition import partition_export
from tests.shadowspill.pytorch.lowering.test_lowering import _measurement


def test_literals_between_tensor_outputs_preserve_public_positions():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 2))

        def forward(self, x):
            y = x @ self.weight
            return None, y, {"status": "ok", "loss": y.sum(), "optional": None}

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_device_model(Model(), mode)
    inputs = fake_device_inputs([torch.randn(4, 3)], mode)
    with mode, torch.no_grad():
        captured = capture_forward(model, inputs)
        partitioned = partition_export(captured, model, partition="whole")
        artifacts = capture_forward_stage_artifacts(partitioned)
    lowered = lower_partitioned_forward_program(
        model,
        partitioned,
        artifacts,
        tuple(_measurement(artifact) for artifact in artifacts),
    )
    assert len(lowered.public_outputs) == 5
    assert [
        i for i, value in enumerate(lowered.public_outputs) if value is not None
    ] == [1, 3]


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
