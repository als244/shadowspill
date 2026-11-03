"""Long authentic controls survive the standalone build/profile boundary."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.build.metadata import _input_source
from shadowspill.errors import CaptureError
from shadowspill.execution.host_values import HostValues
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance, TensorGeometry
from shadowspill.pytorch.task_inputs import _initialize
from shadowspill.task.inputs import TaskInputRole


def test_large_noncontiguous_control_sample_round_trips_exactly():
    source = torch.arange(2 * 131_073, dtype=torch.int64).view(2, -1).T
    with FakeTensorMode(allow_non_fake_inputs=True):
        record = _input_source(
            TaskInputProvenance(TaskInputRole.CONTROL, representative_value=source)
        )
    assert "value_bytes" in record and "value" not in record
    restored = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype)
    _initialize(restored, record)
    torch.testing.assert_close(restored, source, rtol=0, atol=0)
    runtime = HostValues._constant(
        TensorGeometry.from_tensor(source).identity(), record
    )
    torch.testing.assert_close(runtime, source, rtol=0, atol=0)
    with pytest.raises(CaptureError, match="byte count"):
        _initialize(torch.empty(1, dtype=source.dtype), record)


def test_representative_parameters_are_not_serialized_as_control_samples():
    source = torch.randn(262_144)
    record = _input_source(
        TaskInputProvenance(TaskInputRole.PARAMETER, representative_value=source)
    )
    assert not {"value", "value_bytes"} & record.keys()
