from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch import InputGuardError, forward_pass


class _Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(128, 128, bias=False) for _ in range(2)])

    def forward(self, value: torch.Tensor, width: int) -> tuple[torch.Tensor, ...]:
        for layer in self.layers:
            value = torch.relu(layer(value))
        return (value[:, :width],)


@pytest.mark.cuda
def test_public_forward_executes_reloads_and_restores(tmp_path: object) -> None:
    if "SHADOWSPILL_PYTORCH_LIBRARY" not in os.environ:
        pytest.skip("the built PyTorch adapter was not provided")
    os.environ["SHADOWSPILL_PROFILE_CACHE"] = str(tmp_path)
    torch.manual_seed(19)
    model = _Network().eval()
    reference = _Network().eval()
    reference.load_state_dict(model.state_dict())
    parameter_ids = tuple(id(value) for value in model.parameters())
    inputs = torch.randn(3, 128)

    planned = forward_pass(
        model,
        example_inputs=[inputs, 17],
        device_budget=2 << 30,
        host_budget=1 << 30,
    )
    assert planned.plan_report.mode == "forward"
    assert planned.plan_report.predicted_makespan_ns > 0
    assert planned.plan_report.predicted_device_peak_bytes == 2 << 30
    assert planned.plan_report.capture_identity
    actual = planned([inputs, 17])[0]
    torch.testing.assert_close(
        actual.cpu(), reference(inputs, 17)[0], rtol=2e-5, atol=2e-6
    )

    snapshot = planned.state_dict()
    planned.load_state_dict(
        {name: torch.zeros_like(value) for name, value in snapshot.items()}
    )
    assert torch.count_nonzero(planned([inputs, 17])[0]).item() == 0
    planned.load_state_dict(snapshot)
    with pytest.raises(InputGuardError):
        planned([inputs, 16])
    planned.close()
    planned.close()

    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    assert all(value.device.type == "cpu" for value in model.parameters())
    torch.testing.assert_close(
        actual.cpu(), reference(inputs, 17)[0], rtol=2e-5, atol=2e-6
    )
    with pytest.raises(RuntimeError, match="closed"):
        planned([inputs, 17])
