"""Both public training builders validate retention before device preparation."""

import pytest
import torch
from torch import nn

from shadowspill import build_forward_backward, build_step


@pytest.mark.parametrize("builder", [build_forward_backward, build_step])
def test_invalid_graph_pair_policy_is_refused_before_preparation(builder):
    model = nn.Linear(2, 2, device="meta")
    options = (
        {"optimizer": torch.optim.SGD(model.parameters(), lr=0.01)}
        if builder is build_step
        else {}
    )
    with pytest.raises(TypeError, match="graph_pair_policy must be a GraphPairPolicy"):
        builder(
            model,
            objective=lambda model, x: model(x).sum(),
            graph_pair_policy=object(),
            **options,
        )
