"""Both public training builders validate retention before device preparation."""

import pytest
import torch
from torch import nn

from shadowspill import build_forward_backward, build_step
from shadowspill.errors import CaptureError
from shadowspill.pytorch import GradientDestinationRule


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


@pytest.mark.parametrize("builder", [build_forward_backward, build_step])
@pytest.mark.parametrize("duplicate", [False, True])
def test_invalid_destination_rules_are_refused_before_device_access(builder, duplicate):
    model = nn.Linear(2, 2, device="meta")
    options = (
        {"optimizer": torch.optim.SGD(model.parameters(), lr=0.01)}
        if builder is build_step
        else {}
    )
    rule = GradientDestinationRule(
        "provider.fresh.default", "provider.accumulating.default", (0,), 1
    )
    error, message = (
        (CaptureError, "repeat an operator")
        if duplicate
        else (TypeError, "GradientDestinationRule objects")
    )
    with pytest.raises(error, match=message):
        builder(
            model,
            objective=lambda model, x: model(x).sum(),
            gradient_destination_rules=(rule, rule) if duplicate else (object(),),
            **options,
        )
