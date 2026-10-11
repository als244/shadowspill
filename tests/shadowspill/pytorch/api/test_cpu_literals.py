"""Inline CPU constants keep their residency through both public planners."""

import copy
from contextlib import ExitStack
from functools import partial

import pytest
import torch
from torch import nn

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import (
    ObjectiveResult,
    Runtime,
    import_model_state,
    plan_forward,
    plan_step,
    release_model_state,
)
from shadowspill.ssd import ssd

pytestmark = [pytest.mark.cuda, pytest.mark.fresh_process]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, -0.25]))

    def forward(self, x):
        bounds = torch.tensor([0, x.shape[0]], device="cpu", dtype=torch.int64)
        return bounds[1:].clone(), x * self.weight


def _objective(model, x):
    bounds, output = model(x)
    return ObjectiveResult(output.square().sum(), {"bounds": bounds})


def _runtime(stack, directory):
    runtime = stack.enter_context(
        Runtime(
            pools={
                "execution": device(physical_capacity=2 << 30),
                "spill": ssd(capacity=512 << 20, directory=directory),
            },
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
            calibrate=False,
        )
    )
    runtime.calibrate_transfer_capabilities(
        large_copy_bytes=4 << 20, warmup_copies=1, measured_copies=2
    )
    return runtime


def test_cpu_literal_forward_from_ssd(tmp_path):
    model = _Model().requires_grad_(False)
    reference = copy.deepcopy(model)
    with ExitStack() as stack:
        runtime = _runtime(stack, tmp_path)
        model = import_model_state(model, runtime=runtime, pool="spill")
        stack.callback(release_model_state, model, runtime=runtime)
        forward = stack.enter_context(
            plan_forward(
                model,
                forward_fn=lambda module, x: module(x),
                example_inputs=[torch.ones(3, 2)],
                runtime=runtime,
                execution="execution",
                spill="spill",
                artifact_store=tmp_path / "artifacts",
                profiling_options=CORRECTNESS_PROFILING,
            )
        )
        for value in (1.0, -2.0, 3.0):
            x = torch.full((3, 2), value)
            bounds, output = forward([x])
            expected_bounds, expected = reference(x)
            assert bounds.device.type == "cpu"
            torch.testing.assert_close(bounds, expected_bounds)
            torch.testing.assert_close(output.cpu(), expected)
            del bounds, output


def test_cpu_literal_training_from_ssd(tmp_path):
    model = _Model()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    examples = [[torch.ones(3, 2)], [torch.ones(5, 2)]]
    with ExitStack() as stack:
        runtime = _runtime(stack, tmp_path)
        model = import_model_state(model, runtime=runtime, pool="spill")
        stack.callback(release_model_state, model, runtime=runtime)
        training = stack.enter_context(
            plan_step(
                model,
                objective=_objective,
                optimizer=partial(torch.optim.SGD, lr=0.01, foreach=False),
                example_inputs=examples,
                runtime=runtime,
                execution="execution",
                spill="spill",
                artifact_store=tmp_path / "artifacts",
                profiling_options=CORRECTNESS_PROFILING,
            )
        )
        for value in (0.5, -1.0, 1.5):
            batches = [[x * value] for (x,) in examples]
            optimizer.zero_grad(set_to_none=True)
            expected_losses = []
            for (x,) in batches:
                result = _objective(reference, x)
                result.loss.backward()
                expected_losses.append(result.loss.detach())
            optimizer.step()
            actual = training(batches)
            for loss, expected in zip(actual.objectives, expected_losses, strict=True):
                torch.testing.assert_close(loss.cpu(), expected)
            for metric, (x,) in zip(actual.metrics, batches, strict=True):
                assert metric["bounds"].device.type == "cpu"
                torch.testing.assert_close(metric["bounds"], torch.tensor([len(x)]))
            state = training.state_dict()["model"]
            torch.testing.assert_close(state["weight"], reference.weight.detach())
            del actual, state, loss
