"""Both checkpoint weight choices restore the requested compute/master state."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from shadowspill.training import checkpoints
from shadowspill.training.observations import MetricSummary
from workloads.recipes.text.reference import checkpoint
from workloads.recipes.text.reference.engine import EagerStep


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([[0.123, -0.235]], dtype=torch.bfloat16)
        )
        self.alias = self.weight
        self.register_buffer("count", torch.tensor(4))

    def forward(self, x):
        return x @ self.weight.t()


@pytest.mark.parametrize("weights", ["master", "compute"])
@pytest.mark.parametrize("restore_master", [False, True])
@pytest.mark.parametrize("include_optimizer", [False, True])
def test_single_weight_representation_and_casts(
    tmp_path, weights, restore_master, include_optimizer
):
    model = Tied()
    master = nn.Parameter(model.weight.detach().float() + 0.0001)
    optimizer = torch.optim.AdamW([master], lr=0.01)
    master.grad = torch.ones_like(master)
    optimizer.step()
    path = tmp_path / "state.pt"
    checkpoints.save(
        path,
        model,
        optimizer,
        7,
        {"weight": master},
        weights=weights,
        include_optimizer=include_optimizer,
    )
    state = torch.load(path, weights_only=True)
    saved = master.detach() if weights == "master" else model.weight.detach()
    assert set(state) == {"model", "step"} | (
        {"optimizer"} if include_optimizer else set()
    )
    assert state["model"]["weight"].dtype == saved.dtype
    torch.testing.assert_close(state["model"]["weight"], saved, rtol=0, atol=0)
    assert state["model"]["weight"].data_ptr() == state["model"]["alias"].data_ptr()
    target = Tied()
    restored_master = nn.Parameter(torch.zeros_like(master))
    target_optimizer = torch.optim.AdamW(
        [restored_master if restore_master else target.weight]
    )
    # Loading parameter-only state must discard old moments, not silently reuse them.
    target_parameter = target_optimizer.param_groups[0]["params"][0]
    target_parameter.grad = torch.ones_like(target_parameter)
    target_optimizer.step()
    assert (
        checkpoints.load(
            state,
            target,
            target_optimizer,
            {"weight": restored_master} if restore_master else None,
        )
        == 7
    )
    torch.testing.assert_close(target.weight, saved.bfloat16(), rtol=0, atol=0)
    assert target.alias is target.weight
    if restore_master:
        torch.testing.assert_close(restored_master, saved.float(), rtol=0, atol=0)
    if restore_master and include_optimizer:
        torch.testing.assert_close(
            target_optimizer.state[restored_master]["exp_avg"],
            optimizer.state[master]["exp_avg"],
            rtol=0,
            atol=0,
        )
    if not include_optimizer:
        assert not target_optimizer.state


class Source:
    name = "sample"

    def __init__(self):
        self.position = 1

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


@pytest.mark.parametrize("weights", ["master", "compute"])
@pytest.mark.parametrize("include_optimizer", [False, True])
def test_reference_checkpoint_restores_requested_weight_and_optimizer_state(
    tmp_path, weights, include_optimizer
):
    sample = torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16)
    options = {
        "device": "cpu",
        "compile": False,
        "objective": lambda model, x: model(x).float().square().sum(),
        "optimizer": torch.optim.AdamW,
        "optimizer_args": {"lr": 0.01},
        "master_dtype": torch.float32,
        "checkpoint_weights": weights,
        "checkpoint_include_optimizer": include_optimizer,
    }
    engine = EagerStep(Tied(), options, initialize=None)
    engine.step([(sample, 1.0)], 1, lambda _: MetricSummary())
    before = engine.weights.masters["weight"].detach().clone()
    path = tmp_path / "step-one"
    checkpoint.save(path, engine, Source(), 1, 0.25, options)
    saved = torch.load(path / "state.pt", weights_only=True)["model"]["weight"]
    assert saved.dtype == (torch.float32 if weights == "master" else torch.bfloat16)

    resumed = EagerStep(Tied(), options, initialize=None)
    source = Source()
    source.position = 0
    assert checkpoint.load(path, resumed, source) == (1, 0.25)
    assert source.position == 1 and resumed.model.alias is resumed.model.weight
    expected = before if weights == "master" else saved.float()
    torch.testing.assert_close(
        resumed.weights.masters["weight"], expected, rtol=0, atol=0
    )
    resumed.step([(sample, 1.0)], 2, lambda _: MetricSummary())
    parameter = resumed.weights.masters["weight"]
    assert resumed.optimizer.state[parameter]["step"].item() == (
        2 if include_optimizer else 1
    )
    # Reloading an optimizer-free checkpoint discards the moments just created.
    if not include_optimizer:
        checkpoint.load(path, resumed, source)
        assert not resumed.optimizer.state


def test_invalid_choice_does_not_write_checkpoint(tmp_path):
    model = Tied()
    with pytest.raises(ValueError, match="weights"):
        checkpoints.save(
            tmp_path / "bad.pt",
            model,
            torch.optim.SGD(model.parameters()),
            0,
            weights="both",
        )
    assert not (tmp_path / "bad.pt").exists()
