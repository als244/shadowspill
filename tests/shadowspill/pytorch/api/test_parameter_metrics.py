"""Once-per-step observations are raw planned outputs, before optimizer writes."""

import copy
from functools import partial

import pytest
import torch

from shadowspill.pytorch import plan_step
from shadowspill.training.observations import parameter_norms
from qualification.profiling import CORRECTNESS_PROFILING

from ..runtime_test_support import public_test_runtime
from .test_01_public_training import (
    _require_adapter,
    _training_objective,
    _TrainingNetwork,
)


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_parameter_metrics_depth_first(tmp_path):
    _check_parameter_metrics(tmp_path, breadth=1)


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_parameter_metrics_breadth_first(tmp_path):
    _check_parameter_metrics(tmp_path, breadth=2)


def _check_parameter_metrics(tmp_path, breadth):
    _require_adapter()
    torch.manual_seed(142)
    model = _TrainingNetwork()
    reference = copy.deepcopy(model)
    batches = [
        [torch.randn(rows, 6), torch.randn(rows, 3), tag]
        for rows, tag in ((2, "left"), (4, "right"))
    ]
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.02, foreach=False)
    expected = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            _training_objective(reference, *batch).loss.backward()
        expected.append(
            {
                name: parameter_norms(weight, weight.grad)
                for name, weight in reference.named_parameters()
            }
        )
        optimizer.step()

    step = plan_step(
        model,
        objective=_training_objective,
        optimizer=partial(torch.optim.SGD, lr=0.02, foreach=False),
        parameter_metrics=parameter_norms,
        example_inputs=batches,
        runtime=public_test_runtime(),
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
        depth=2 // breadth,
        breadth=breadth,
        profiling_options=CORRECTNESS_PROFILING,
    )
    retained = []
    for index in range(2):
        result = step(batches)
        assert tuple(value["tag"] for value in result.metrics) == ("left", "right")
        assert set(result.parameter_metrics) == set(expected[index])
        for name, metrics in result.parameter_metrics.items():
            for key, value in metrics.items():
                assert value.device.type == "cuda" and value.dtype == torch.float32
                assert not value.requires_grad
                torch.testing.assert_close(
                    value.cpu(), expected[index][name][key], rtol=2e-5, atol=2e-6
                )
        retained.append(result)
    # A later optimizer write/invocation cannot overwrite earlier observations.
    for name, metrics in retained[0].parameter_metrics.items():
        for key, value in metrics.items():
            torch.testing.assert_close(
                value.cpu(), expected[0][name][key], rtol=2e-5, atol=2e-6
            )
    state = step.state_dict()["model"]
    for name, value in reference.state_dict().items():
        if name in state:
            torch.testing.assert_close(state[name], value, rtol=2e-5, atol=2e-6)
    step.close()
