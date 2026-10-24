"""The checkpoint format: a master written in its weight's place, and restored
exactly."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from shadowspill.training import checkpoints
from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import PyTorch


def _bf16_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(4, 3).to(torch.bfloat16)


def test_a_weight_with_a_master_is_written_as_its_master(tmp_path: Path) -> None:
    model = _bf16_model(0)
    # Not the weights' cast exactly, as a master rarely is once it has stepped.
    masters = {
        name: nn.Parameter(weight.detach().float() + 1e-3)
        for name, weight in model.named_parameters()
    }
    path = tmp_path / "checkpoint.pt"
    checkpoints.save(
        path, model, torch.optim.AdamW(masters.values()), step=7, masters=masters
    )
    state = torch.load(path, weights_only=True)
    assert set(state) == {"model", "optimizer", "step"}
    for name, master in masters.items():
        assert torch.equal(state["model"][name], master.detach())

    restored = _bf16_model(1)
    restored_masters = {
        name: nn.Parameter(torch.zeros_like(master)) for name, master in masters.items()
    }
    optimizer = torch.optim.AdamW(restored_masters.values())
    assert checkpoints.load(state, restored, optimizer, restored_masters) == 7
    for name, master in masters.items():
        assert torch.equal(restored_masters[name].detach(), master.detach())
        assert torch.equal(restored.state_dict()[name], master.detach().bfloat16())


def test_a_model_without_masters_is_written_as_it_is(tmp_path: Path) -> None:
    model = _bf16_model(0)
    path = tmp_path / "checkpoint.pt"
    checkpoints.save(path, model, torch.optim.AdamW(model.parameters()), step=1)
    state = torch.load(path, weights_only=True)

    restored = _bf16_model(1)
    optimizer = torch.optim.AdamW(restored.parameters())
    assert checkpoints.load(state, restored, optimizer) == 1
    for name, value in model.state_dict().items():
        assert torch.equal(state["model"][name], value)
        assert torch.equal(restored.state_dict()[name], value)


class PartlyTrained(nn.Module):
    def __init__(self, *, device="cpu"):
        super().__init__()
        self.base = nn.Parameter(
            torch.ones(3, 2, dtype=torch.bfloat16, device=device), requires_grad=False
        )
        self.delta = nn.Parameter(
            torch.zeros(3, 2, dtype=torch.bfloat16, device=device)
        )
        self.moving = nn.Parameter(
            torch.tensor(0.01, device=device), requires_grad=False
        )
        self.register_buffer("calls", torch.tensor(0.0, device=device))

    def forward(self, x):
        with torch.no_grad():
            self.moving.add_(0.001)
            self.calls.add_(1)
        return (x @ (self.base + self.delta)).float() + self.moving + self.calls


def test_trainer_partial_checkpoint_restores_masters_and_mutated_frozen_state(tmp_path):
    batch = torch.ones(2, 3, dtype=torch.bfloat16)
    initial = PartlyTrained().state_dict()

    def trainer(*, device="cpu", identity="base-v1"):
        return Trainer(
            PartlyTrained(device=device),
            objective=lambda model, data: model(data).square().mean(),
            optimizer=torch.optim.AdamW,
            optimizer_args={"lr": 0.01},
            master_dtype=torch.float32,
            grad_dtype=torch.float32,
            frozen_state_id=identity,
            backend=PyTorch(device="cpu", compile=False),
        )

    first = trainer().prepare(batch)
    first.step(batch)
    saved_path = first.save(tmp_path / "step-one")
    saved = torch.load(saved_path / "state.pt", weights_only=True)
    assert set(saved["model"]) == {"delta", "moving", "calls"}
    assert saved["model"]["delta"].dtype == torch.float32
    assert len(saved["optimizer"]["state"]) == 1
    first.step(batch)
    expected = {
        name: tensor.clone() for name, tensor in first.model.state_dict().items()
    }
    first.close()

    with pytest.raises(ValueError, match="frozen_state_id"):
        trainer(identity="another-base").prepare(batch, checkpoint=saved_path)
    with pytest.raises(ValueError, match="initialize="):
        trainer(device="meta").prepare(batch, checkpoint=saved_path)
    resumed = trainer(device="meta")
    resumed.prepare(
        batch,
        initialize=lambda model: model.load_state_dict(initial),
        checkpoint=saved_path,
    )
    resumed.step(batch)
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    again = resumed.save(tmp_path / "step-two")
    assert set(torch.load(again / "state.pt", weights_only=True)["model"]) == {
        "delta",
        "moving",
        "calls",
    }
    with pytest.raises(ValueError, match="trainable parameter"):
        checkpoints.load(
            {
                **saved,
                "model": {k: v for k, v in saved["model"].items() if k != "delta"},
            },
            resumed.model,
            resumed._execution.optimizer,
            resumed._execution.weights.masters,
            frozen_state_id="base-v1",
        )
    resumed.close()
