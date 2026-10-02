"""Read-only observations of completed gradients retain independent updates."""

import pytest
import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.optimizer.metrics import with_parameter_metrics


def norm(parameter, gradient):
    return {
        "param_norm": torch.linalg.vector_norm(parameter, dtype=torch.float32),
        "grad_norm": torch.linalg.vector_norm(gradient, dtype=torch.float32),
    }


def test_gradient_observations_precede_each_update_and_use_accumulated_dtype():
    params = {
        name: torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        for name in ("left", "right")
    }
    for param in params.values():
        param.grad = torch.zeros_like(param)
    optimizer = torch.optim.SGD(params.values(), lr=0.1, foreach=False)
    captured = capture_optimizer(params, optimizer, gradient_dtype=torch.float32)
    instrumented = with_parameter_metrics(captured, norm)
    assert len(instrumented.update_tasks) == 2 * len(captured.update_tasks)
    names = set()
    for observation, update in zip(
        instrumented.update_tasks[::2], instrumented.update_tasks[1::2], strict=True
    ):
        assert observation.metric_schema is not None
        assert update.metric_schema is None
        assert not observation.mutation_names
        assert not observation.artifact.storage_contract.mutations
        assert not any(
            "_local_scalar_dense" in op or "synchronize" in op
            for op in observation.artifact.operator_targets
        )
        assert [value.dtype for value in observation.artifact.example_arguments] == [
            torch.bfloat16,
            torch.float32,
        ]
        values = tuple(
            torch.tensor([3.0, 4.0, 0.0, 0.0]) for _ in observation.binding_names
        )
        output = observation.artifact.graph_module(*values)
        metrics = observation.metric_schema.rebuild_metrics(output)
        for name, metric in metrics.items():
            assert metric["grad_norm"].item() == 5.0
            assert metric["param_norm"].item() == 5.0
            names.add(name)
    assert names == set(params)
    assert with_parameter_metrics(captured, None) is captured


def test_gradient_observer_cannot_mutate_input():
    param = torch.nn.Parameter(torch.ones(4))
    param.grad = torch.ones_like(param)
    captured = capture_optimizer({"weight": param}, torch.optim.SGD([param], lr=0.1))

    def mutating(parameter, gradient):
        return gradient.zero_().sum()

    with pytest.raises(CaptureError, match="must not mutate"):
        with_parameter_metrics(captured, mutating)


def test_observations_use_compute_weights_with_masters_and_keep_two_raw_tensors():
    master = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    master.grad = torch.zeros_like(master)
    compute = torch.ones(4, dtype=torch.bfloat16)
    captured = capture_optimizer(
        {"weight": master},
        torch.optim.SGD([master], lr=0.1),
        compute_copies={"weight": compute},
        gradient_dtype=torch.bfloat16,
    )
    observation = with_parameter_metrics(captured, norm).update_tasks[0]
    assert observation.binding_names == ("compute.weight", "gradient.weight")
    result = observation.artifact.graph_module(compute, torch.full_like(compute, 3))
    assert len(result) == 2
    assert all(
        isinstance(value, torch.Tensor) and value.dtype == torch.float32
        for value in result
    )
    metrics = observation.metric_schema.rebuild_metrics(result)["weight"]
    assert metrics["param_norm"].item() == 2
    assert metrics["grad_norm"].item() == 6


def test_returned_views_are_snapshots_before_optimizer_mutation():
    parameter = torch.nn.Parameter(torch.ones(4))
    parameter.grad = torch.ones_like(parameter)
    captured = capture_optimizer(
        {"weight": parameter}, torch.optim.SGD([parameter], lr=0.1)
    )
    observation = with_parameter_metrics(
        captured, lambda weight, grad: {"first": weight[0]}
    ).update_tasks[0]
    values = observation.artifact.graph_module(parameter.detach(), parameter.grad)
    with torch.no_grad():
        parameter.zero_()
    assert (
        observation.metric_schema.rebuild_metrics(values)["weight"]["first"].item() == 1
    )


def test_observations_capture_on_the_selected_process_device():
    parameter = torch.nn.Parameter(torch.ones(4))
    parameter.grad = torch.ones_like(parameter)
    captured = capture_optimizer(
        {"weight": parameter}, torch.optim.SGD([parameter], lr=0.1)
    )
    observation = with_parameter_metrics(captured, norm, device_index=3).update_tasks[0]
    assert all(
        value.device == torch.device("cuda:3")
        for value in observation.artifact.example_arguments
    )
