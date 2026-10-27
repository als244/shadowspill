"""Equivalent example signatures reuse export without sharing runtime values."""

import pytest
import torch
from torch import nn

from shadowspill.build import training_capture


@pytest.mark.parametrize("mixed", [False, True])
def test_one_export_per_input_signature(monkeypatch, mixed):
    if not torch.cuda.is_available():
        pytest.skip("fake CUDA export needs a CUDA-enabled build")
    calls = [((torch.zeros(2 + (mixed and i % 2), 4),), {}) for i in range(4)]
    original = training_capture.capture_training_objective
    exports = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        exports.append(result)
        return result

    monkeypatch.setattr(training_capture, "capture_training_objective", capture)
    captured = training_capture.capture_training_model(
        nn.Linear(4, 4),
        lambda model, x: model(x).sum(),
        (),
        {},
        num_examples=1,
        example_calls=calls,
        fuse_narrow_gradient_accumulation=True,
        partition="whole",
        operator_recompute=None,
        gradient_dtype=None,
        device_index=0,
    )
    assert len(exports) == 1 + mixed
    assert len(captured.examples) == 4
    inputs = [item.training.exported.flat_inputs for item in captured.examples]
    assert len({id(value) for value in inputs}) == 4
    assert len({s.digest for s in captured.input_signatures}) == 1 + mixed
