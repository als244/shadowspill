"""Legacy tuple controls still validate arity before updating any group."""

import pytest
import torch
from torch import nn

from shadowspill.pytorch.callables import _apply_hyperparams


def test_tuple_hyperparameters_broadcast_to_each_group():
    groups = [{"betas": (torch.tensor(0.9), torch.tensor(0.99))} for _ in range(2)]
    _apply_hyperparams(nn.Linear(1, 1), groups, {"betas": (0.8, 0.95)}, lambda _: None)
    for group in groups:
        assert [value.item() for value in group["betas"]] == pytest.approx([0.8, 0.95])
    with pytest.raises(ValueError, match="per group"):
        _apply_hyperparams(nn.Linear(1, 1), groups, {"betas": (0.7,)}, lambda _: None)
    for group in groups:
        assert [value.item() for value in group["betas"]] == pytest.approx([0.8, 0.95])
