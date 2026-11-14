"""The reference runner accepts the same explicit optimizer names as saved Plans."""

import pytest
import torch
from torch import nn

from shadowspill.training._hyperparams import set_hyperparams


def test_named_optimizer_group_only_updates_that_group_in_place():
    model = nn.Linear(2, 2)
    rates = torch.tensor(0.1), torch.tensor(0.2)
    optimizer = torch.optim.SGD(
        [
            {"params": [model.weight], "lr": rates[0]},
            {"params": [model.bias], "lr": rates[1]},
        ]
    )
    set_hyperparams(model, optimizer, {"optimizer_group.1.lr": 0.03})
    assert optimizer.param_groups[0]["lr"] is rates[0]
    assert optimizer.param_groups[1]["lr"] is rates[1]
    assert [rate.item() for rate in rates] == pytest.approx([0.1, 0.03])
    set_hyperparams(model, optimizer, {"lr": 0.04})
    assert [rate.item() for rate in rates] == pytest.approx([0.04, 0.04])
    for invalid in ("optimizer_group.2.lr", "optimizer_group.0", "optimizer_group.0.x"):
        with pytest.raises(KeyError):
            set_hyperparams(model, optimizer, {invalid: 0.1})
