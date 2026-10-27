"""Unsupported initializer and optimizer fallbacks fail before kernel compilation."""

import pytest
import torch
from torch import nn

from shadowspill.build.optimizer import capture_update
from shadowspill.build.training_capture import capture_training_model
from shadowspill.errors import CaptureError
from shadowspill.pytorch.optimizer import trace


def capture(model):
    return capture_training_model(
        model,
        lambda model, x: model(x).square().mean(),
        (torch.empty(2, 6),),
        {},
        num_examples=1,
        example_calls=None,
        fuse_narrow_gradient_accumulation=True,
        partition="whole",
        operator_recompute=None,
        gradient_dtype=None,
        device_index=0,
    )


@pytest.mark.cuda
def test_gradient_initialized_state_has_no_guess_before_first_step():
    if not torch.cuda.is_available():
        pytest.skip("capture requires a representative GPU")
    model = nn.Linear(6, 3, device="meta")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    with pytest.raises(CaptureError, match=r"first-use initializer.*gradient"):
        capture_update(model, optimizer, capture(model), gradient_dtype=None)
    assert not optimizer.state and all(p.is_meta for p in model.parameters())


@pytest.mark.cuda
def test_portable_build_refuses_python_optimizer_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("capture requires a representative GPU")

    def reject_graph(_optimizer, _written):
        raise RuntimeError("optimizer graph intentionally unavailable")

    monkeypatch.setattr(trace, "_export_optimizer_graph", reject_graph)
    model = nn.Linear(6, 3, device="meta")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    with pytest.raises(
        CaptureError, match="must be capturable as a tensor-only graph"
    ) as error:
        capture_update(model, optimizer, capture(model), gradient_dtype=None)
    assert "optimizer graph intentionally unavailable" in str(error.value)
    assert not optimizer.state
