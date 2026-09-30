from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from shadowspill.errors import (
    InputGuardError,
)
from shadowspill.planner.quantization import GIBIBYTE
from shadowspill.pytorch import (
    export_model_state,
    import_model_state,
    plan_forward,
)
from shadowspill.runtime.configuration import adapter_path
from tools.qualification.profiling import CORRECTNESS_PROFILING

from ..runtime_test_support import public_test_runtime


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
@pytest.mark.fresh_process
def test_public_forward_executes_reloads_and_restores(tmp_path: object) -> None:
    if torch.cuda.is_initialized():
        pytest.skip("public allocator installation requires a fresh process")
    try:
        adapter_path(None)
    except RuntimeError:
        pytest.skip("the built PyTorch adapter is not installed")
    torch.manual_seed(19)
    model = _Network().eval()
    reference = _Network().eval()
    reference.load_state_dict(model.state_dict())
    inputs = torch.randn(3, 128)
    runtime = public_test_runtime()
    model = import_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    parameter_ids = tuple(id(value) for value in model.parameters())

    planned = plan_forward(
        model,
        profiling_options=CORRECTNESS_PROFILING,
        example_inputs=[inputs, 17],
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
        profiling_metadata={"batch_size": 3, "width": 17},
    )
    assert planned.plan_report.mode == "forward"
    assert planned.plan_report.predicted_makespan_ns > 0
    admission = planned.plan_report.execution_plan.admission
    assert planned.plan_report.execution_budget_bytes == admission.slab_bytes
    assert planned.plan_report.predicted_device_peak_bytes == (
        admission.baseline_bytes
        + admission.external_headroom_bytes
        + admission.slab_bytes
    )
    # The default execution budget is the pool's capacity at whole-GiB
    # granularity, so the plan leaves less than one GiB of the device budget
    # unused.
    assert (
        planned.plan_report.predicted_device_peak_bytes <= admission.device_budget_bytes
    )
    assert (
        admission.device_budget_bytes - planned.plan_report.predicted_device_peak_bytes
        < GIBIBYTE
    )
    assert planned.plan_report.capture_identity
    assert planned.plan_report.program is planned.plan_report.execution_plan.program
    assert planned.plan_report.search_result.program == planned.plan_report.program
    assert planned.plan_report.diagnostics.cache_artifacts
    assert len(planned.plan_report.diagnostics.profiling_metadata) == 1
    assert len(planned.plan_report.diagnostics.physical_layouts) == 1
    layout = planned.plan_report.diagnostics.physical_layouts[0]
    assert layout.plan_role == "forward"
    assert layout.strategy == "fixed"
    assert layout.required_bytes <= layout.pool_capacity_bytes
    assert layout.attempts[-1].accepted
    assert all(item.search_wall_time_ns > 0 for item in layout.attempts)
    assert all(item.physical_admission_wall_time_ns > 0 for item in layout.attempts)
    assert layout.task_memory_envelopes
    encoded_layout = planned.plan_report.diagnostics.as_dict()["physical_layouts"][0]
    assert encoded_layout["attempts"][-1]["search_wall_time_ns"] > 0
    assert encoded_layout["attempts"][-1]["physical_admission_wall_time_ns"] > 0
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
    export_model_state(model, runtime=runtime, release_runtime=True)

    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    assert all(value.device.type == "cpu" for value in model.parameters())
    torch.testing.assert_close(
        actual.cpu(), reference(inputs, 17)[0], rtol=2e-5, atol=2e-6
    )
    with pytest.raises(RuntimeError, match="closed"):
        planned([inputs, 17])


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_forward_sets_a_model_buffer_each_call(tmp_path: object) -> None:
    """A persistent buffer is set per call and read by the forward: the value
    reaches the pool the state lives in, not the module's handle. A buffer
    that is not state, and a name that is not a buffer, are refused."""

    if torch.cuda.is_initialized():
        pytest.skip("public allocator installation requires a fresh process")
    try:
        adapter_path(None)
    except RuntimeError:
        pytest.skip("the built PyTorch adapter is not installed")

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(16, 16, bias=False)
            self.register_buffer("scale", torch.tensor(1.0))
            self.register_buffer("runtime_scale", torch.tensor(1.0), persistent=False)

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return (torch.relu(self.layer(value)) * self.scale * self.runtime_scale,)

    torch.manual_seed(23)
    model = Network().eval()
    reference = Network().eval()
    reference.load_state_dict(model.state_dict())
    inputs = torch.randn(3, 16)
    runtime = public_test_runtime()
    model = import_model_state(
        model, runtime=runtime, pool="spill", release_source=True
    )
    planned = plan_forward(
        model,
        profiling_options=CORRECTNESS_PROFILING,
        example_inputs=[inputs],
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    for scale in (2.0, 0.5, 3.0):
        actual = planned([inputs], hyperparams={"scale": scale})[0]
        with torch.no_grad():
            reference.scale.fill_(scale)
        torch.testing.assert_close(
            actual.cpu(), reference(inputs)[0], rtol=2e-5, atol=2e-6
        )
    assert planned.state_dict()["scale"].item() == 3.0
    with pytest.raises(KeyError, match="persistent"):
        planned([inputs], hyperparams={"runtime_scale": 2.0})
    with pytest.raises(KeyError, match="no model buffer"):
        planned([inputs], hyperparams={"lr": 1.0})
    planned.close()
    export_model_state(model, runtime=runtime, release_runtime=True)
    assert model.scale.item() == 3.0
