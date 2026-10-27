"""Ordinary reference checkpoints preserve quantized components and logical masters."""

import pytest
import torch

from shadowspill.training import checkpoints
from shadowspill.training._model import initialize_model
from shadowspill.training.observations import MetricSummary
from tests.shadowspill.pytorch.representation_fixtures import model
from workloads.recipes.text.reference.engine import EagerStep


@pytest.mark.parametrize("weights", ["master", "compute"])
def test_wrapper_checkpoint_publishes_or_preserves_compute(tmp_path, weights):
    source = model()
    master = torch.nn.Parameter(source.weight.dense().detach() + 0.004)
    optimizer = torch.optim.SGD([master], lr=0.01)
    path = tmp_path / "state.pt"
    checkpoints.save(path, source, optimizer, 3, {"weight": master}, weights=weights)
    state = torch.load(path, weights_only=True)
    target = model()
    restored_master = torch.nn.Parameter(torch.zeros_like(master))
    restored_optimizer = torch.optim.SGD([restored_master], lr=0.01)
    checkpoints.load(state, target, restored_optimizer, {"weight": restored_master})
    expected = (
        master.detach() if weights == "master" else source.weight.dense().detach()
    )
    torch.testing.assert_close(restored_master, expected, rtol=0, atol=0)
    reference = source.weight.detach().clone()
    if weights == "master":
        reference.copy_(master.detach())
    torch.testing.assert_close(target.weight.payload, reference.payload, rtol=0, atol=0)
    torch.testing.assert_close(target.weight.scale, reference.scale, rtol=0, atol=0)
    with torch.device("meta"):
        initialized = model()
    initialize_model(initialized, state=state["model"])
    torch.testing.assert_close(
        initialized.weight.payload, reference.payload, rtol=0, atol=0
    )


@pytest.mark.parametrize("weights", ["master", "compute"])
def test_reference_updates_logical_master_and_resumes(tmp_path, weights):
    sample = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24

    def engine():
        return EagerStep(
            model(),
            dict(
                objective=lambda net, data: net(data).square().mean(),
                optimizer=torch.optim.SGD,
                optimizer_args={"lr": 0.01},
                master_dtype=torch.float32,
                device="cpu",
                compile=False,
            ),
            None,
        )

    calls = [(sample, torch.tensor(1.0))]
    original = engine()
    before = original.model.weight.payload.clone()
    for step in range(1, 4):
        original.step(calls, step, lambda _: MetricSummary())
    assert "weight" in original.weights.masters
    assert not torch.equal(before, original.model.weight.payload)
    path = tmp_path / "checkpoint.pt"
    checkpoints.save(
        path,
        original.model,
        original.optimizer,
        3,
        original.weights.masters,
        weights=weights,
    )
    expected = original.step(calls, 4, lambda _: MetricSummary()).loss
    resumed = engine()
    state = torch.load(path, weights_only=True)
    step = checkpoints.load(
        state, resumed.model, resumed.optimizer, resumed.weights.masters
    )
    result = resumed.step(calls, step + 1, lambda _: MetricSummary())
    assert result.step == 4
    assert result.loss == pytest.approx(expected, rel=1e-6, abs=1e-7)
