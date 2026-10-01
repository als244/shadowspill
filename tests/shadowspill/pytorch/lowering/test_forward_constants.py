"""Static leaves keep their positions without acquiring tensor storage."""

from __future__ import annotations

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture.aot import capture_forward
from shadowspill.pytorch.capture.artifacts import capture_forward_stage_artifacts
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.lowering.forward import lower_partitioned_forward_program
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
