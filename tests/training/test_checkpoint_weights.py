"""Both checkpoint weight choices restore the requested compute/master state."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from shadowspill.training import Trainer, checkpoints
from shadowspill.training.backends import PyTorch


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
def test_single_weight_representation_and_casts(tmp_path, weights, restore_master):
    model = Tied()
    master = nn.Parameter(model.weight.detach().float() + 0.0001)
    optimizer = torch.optim.AdamW([master], lr=0.01)
    master.grad = torch.ones_like(master)
    optimizer.step()
    path = tmp_path / "state.pt"
    checkpoints.save(path, model, optimizer, 7, {"weight": master}, weights=weights)
    state = torch.load(path, weights_only=True)
    saved = master.detach() if weights == "master" else model.weight.detach()
    assert set(state) == {"model", "optimizer", "step"}
    assert state["model"]["weight"].dtype == saved.dtype
    torch.testing.assert_close(state["model"]["weight"], saved, rtol=0, atol=0)
    assert state["model"]["weight"].data_ptr() == state["model"]["alias"].data_ptr()
    target = Tied()
    restored_master = nn.Parameter(torch.zeros_like(master))
    target_optimizer = torch.optim.AdamW(
        [restored_master if restore_master else target.weight]
    )
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
        torch.testing.assert_close(
            target_optimizer.state[restored_master]["exp_avg"],
            optimizer.state[master]["exp_avg"],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("weights", ["master", "compute"])
def test_trainer_periodic_checkpoint_threads_choice(tmp_path, weights):
    sample = torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16)

    def build():
        return Trainer(
            Tied(),
            objective=lambda model, x: model(x).float().square().sum(),
            optimizer=torch.optim.AdamW,
            optimizer_args={"lr": 0.01},
            master_dtype=torch.float32,
            backend=PyTorch(compile=False, device="cpu"),
        )

    with build() as trainer:
        trainer.prepare(sample)
        trainer.fit(
            [sample],
            steps=1,
            run_dir=tmp_path,
            checkpoint_every=1,
            checkpoint_weights=weights,
            log_every=0,
        )
        before = trainer._execution.weights.masters["weight"].detach().clone()
    checkpoint = tmp_path / "checkpoints/step_00000001"
    saved = torch.load(checkpoint / "state.pt", weights_only=True)["model"]["weight"]
    assert saved.dtype == (torch.float32 if weights == "master" else torch.bfloat16)
    with build() as resumed:
        resumed.prepare(sample, checkpoint=checkpoint)
        expected = before if weights == "master" else saved.float()
        torch.testing.assert_close(
            resumed._execution.weights.masters["weight"], expected, rtol=0, atol=0
        )
        assert resumed.step_count == 1


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
