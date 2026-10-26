"""A wrapper's advertised dtype is not its physical optimizer storage."""

import pytest
import torch

from shadowspill.build.optimizer_parameters import local_optimizer_parameters
from shadowspill.pytorch.optimizer.sandbox import copy_optimizer_to_meta
from shadowspill.pytorch.representations import is_wrapper
from tests.shadowspill.pytorch.state.representations import model


@pytest.mark.parametrize("master_dtype", [torch.float32, torch.float64])
def test_dense_master_is_separate_even_when_logical_dtypes_match(master_dtype):
    with torch.device("meta"):
        net = model()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    copied, names = copy_optimizer_to_meta(optimizer, {id(net.weight): "weight"})
    local, names, copies = local_optimizer_parameters(
        copied, names, dict(net.named_parameters()), {"weight"}, master_dtype
    )
    parameter = local.param_groups[0]["params"][0]
    assert parameter.is_meta and parameter.dtype == master_dtype
    assert not is_wrapper(parameter)
    assert names[id(parameter)] == "weight"
    assert copies == {"weight": net.weight}
    assert optimizer.param_groups[0]["params"][0] is net.weight


def test_frozen_wrapper_does_not_acquire_a_master():
    with torch.device("meta"):
        net = model()
    net.weight.requires_grad_(False)
    copied, names = copy_optimizer_to_meta(
        torch.optim.SGD(net.parameters(), lr=0.01), {id(net.weight): "weight"}
    )
    local, _, copies = local_optimizer_parameters(
        copied, names, dict(net.named_parameters()), set(), torch.float32
    )
    assert local is copied and not copies
