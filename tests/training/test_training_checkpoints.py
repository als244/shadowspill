"""The checkpoint format: a master written in its weight's place, and restored
exactly."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training import checkpoints


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
