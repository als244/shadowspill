"""The checkpoint format: each weight written once, and restored exactly."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training import checkpoints


class MasterCopy(torch.optim.Optimizer):
    """Keeps an fp32 master copy of each weight and a moment; steps nothing."""

    def __init__(self, parameters) -> None:
        super().__init__(parameters, {})
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter] = {
                    "master": parameter.detach().float().clone(),
                    "moment": torch.full_like(parameter, 0.5, dtype=torch.float32),
                }

    def step(self, closure=None) -> None:
        return None


def _bf16_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(4, 3).to(torch.bfloat16)


def test_a_weight_its_master_reproduces_is_written_once(tmp_path: Path) -> None:
    model = _bf16_model(0)
    optimizer = MasterCopy(model.parameters())
    path = tmp_path / "checkpoint.pt"
    checkpoints.save(path, model, optimizer, step=7)
    state = torch.load(path, weights_only=True)
    assert state["model"] == {}
    assert state["model_from_optimizer"] == {
        "weight": (0, "master"),
        "bias": (1, "master"),
    }

    restored = _bf16_model(1)
    restored_optimizer = MasterCopy(restored.parameters())
    assert checkpoints.load(state, restored, restored_optimizer) == 7
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_a_weight_no_optimizer_entry_reproduces_is_written_itself(
    tmp_path: Path,
) -> None:
    model = _bf16_model(0)
    optimizer = MasterCopy(model.parameters())
    with torch.no_grad():
        for entries in optimizer.state.values():
            entries["master"] += 1e-3  # no longer the weight's exact source
    path = tmp_path / "checkpoint.pt"
    checkpoints.save(path, model, optimizer, step=1)
    state = torch.load(path, weights_only=True)
    assert set(state["model"]) == {"weight", "bias"}
    assert state["model_from_optimizer"] == {}
