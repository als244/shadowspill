from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch import InputGuardError, plan_forward
from shadowspill.pytorch.runtime_adapter.runtime import _adapter_path

from .runtime_test_support import public_test_runtime


class _Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(128, 128, bias=False) for _ in range(2)])
        self.register_buffer("runtime_scale", torch.tensor(1.0), persistent=False)

    def forward(self, value: torch.Tensor, width: int) -> tuple[torch.Tensor, ...]:
        for layer in self.layers:
            value = torch.relu(layer(value))
        return (value[:, :width] * self.runtime_scale,)


@pytest.mark.cuda
def test_public_forward_executes_reloads_and_restores(tmp_path: object) -> None:
    if torch.cuda.is_initialized():
        pytest.skip("public allocator installation requires a fresh process")
    try:
        _adapter_path(None)
    except RuntimeError:
        pytest.skip("the built PyTorch adapter is not installed")
    torch.manual_seed(19)
    model = _Network().eval()
    reference = _Network().eval()
    reference.load_state_dict(model.state_dict())
    parameter_ids = tuple(id(value) for value in model.parameters())
    inputs = torch.randn(3, 128)

    planned = plan_forward(
        model,
        example_inputs=[inputs, 17],
        runtime=public_test_runtime(),
        execution="execution",
        spill="spill",
        planning_cachedir=tmp_path,
        profiling_metadata={"batch_size": 3, "width": 17},
    )
    assert planned.plan_report.mode == "forward"
    assert planned.plan_report.predicted_makespan_ns > 0
    admission = planned.plan_report.execution_plan.admission
    assert planned.plan_report.execution_budget_bytes == admission.slab_bytes
    assert planned.plan_report.predicted_device_peak_bytes == (
        admission.context_bytes
        + admission.provider_headroom_bytes
        + admission.slab_bytes
    )
    assert (
        planned.plan_report.predicted_device_peak_bytes == admission.device_budget_bytes
    )
    assert planned.plan_report.capture_identity
    assert planned.plan_report.program is planned.plan_report.execution_plan.program
    assert planned.plan_report.pressurefit_result.program == planned.plan_report.program
    assert planned.plan_report.diagnostics.cache_artifacts
    assert len(planned.plan_report.diagnostics.profiling_metadata) == 1
    actual = planned([inputs, 17])[0]
    torch.testing.assert_close(
        actual.cpu(), reference(inputs, 17)[0], rtol=2e-5, atol=2e-6
    )

    snapshot = planned.state_dict()
    assert "runtime_scale" not in snapshot
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
