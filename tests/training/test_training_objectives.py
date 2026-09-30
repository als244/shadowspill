"""Objective state follows a model built on meta, keeping normalization in FP32."""

import pytest
import torch
from torch import nn

from training.objectives import Objective


@pytest.mark.parametrize("buffer_only", [False, True])
def test_meta_objective_materializes_without_mixed_device_state(buffer_only):
    model = nn.Module()
    value = torch.empty(2, device="meta", dtype=torch.float16)
    if buffer_only:
        model.register_buffer("weight", value)
    else:
        model.register_parameter("weight", nn.Parameter(value))
    objective = Objective(model, lambda *_args: torch.tensor(8.0), {})
    assert all(value.is_meta for value in objective.state_dict().values())
    assert objective.trained_total.dtype == torch.float32

    objective.to_empty(device="cpu")
    objective.reset_parameters()
    assert objective.trained_total.item() == 1.0
    objective.trained_total.fill_(4)
    tokens = torch.zeros(1, dtype=torch.int64)
    assert objective(tokens, tokens, tokens).item() == 2.0
