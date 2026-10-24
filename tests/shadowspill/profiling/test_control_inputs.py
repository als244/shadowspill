"""Saved profiling controls retain real values without embedding model weights."""

import json

import pytest
import torch

from shadowspill.build.metadata import _input_source
from shadowspill.errors import CaptureError
from shadowspill.profiling.inputs import _initialize
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.task.inputs import TaskInputRole


def test_long_control_sequence_survives_portable_metadata():
    positions = torch.arange(32768, dtype=torch.int64).reshape(2, 16384) % 8192
    source = _input_source(
        TaskInputProvenance(
            TaskInputRole.CONTROL, "positions", representative_value=positions
        )
    )
    restored = torch.empty_like(positions)
    _initialize(restored, json.loads(json.dumps(source)))
    torch.testing.assert_close(restored, positions, rtol=0, atol=0)


def test_oversized_controls_still_require_explicit_values():
    positions = torch.arange((1 << 20) // 8 + 1)
    source = _input_source(
        TaskInputProvenance(
            TaskInputRole.CONTROL, "positions", representative_value=positions
        )
    )
    with pytest.raises(CaptureError, match="caller-supplied values"):
        _initialize(torch.empty_like(positions), source)


def test_parameter_values_are_not_embedded_in_program():
    source = _input_source(
        TaskInputProvenance(
            TaskInputRole.PARAMETER, "weight", representative_value=torch.ones(8)
        )
    )
    assert "value" not in source
