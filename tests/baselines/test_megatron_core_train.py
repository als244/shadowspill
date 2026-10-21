"""The Megatron trainer's first-update evidence: what the optimizer steps is
found in each of Megatron's optimizer layouts, and checked against the
contract."""

import types

import pytest
import torch
from torch import nn

from baselines.common.precision import Precision
from baselines.megatron_core.train import (
    pinned_parameters,
    precision_evidence,
    stepped_of,
)


def optimizer(groups, states=None):
    """A torch optimizer's surface: its groups and its state by value."""

    return types.SimpleNamespace(
        param_groups=[{"params": list(groups)}], state=dict(states or {})
    )


def moments(values, dtype=torch.float32):
    return {
        id(value): {
            "exp_avg": torch.zeros(value.numel(), dtype=dtype),
            "exp_avg_sq": torch.zeros(value.numel(), dtype=dtype),
        }
        for value in values
    }


class Layer(nn.Module):
    def __init__(self, dtype, pinned=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4, dtype=dtype))
        if pinned is not None:
            self.A_log = nn.Parameter(torch.zeros(4, dtype=pinned))


def test_the_stepped_values_are_the_optimizers_groups():
    copies = [torch.zeros(8)]
    part = types.SimpleNamespace(optimizer=optimizer(copies))
    assert stepped_of(part) == copies


def test_transformer_engines_masters_are_its_state():
    weight = torch.zeros(8, dtype=torch.bfloat16)
    master = torch.zeros(8)
    inner = optimizer([weight], {id(weight): {"master_param": master}})
    assert stepped_of(types.SimpleNamespace(optimizer=inner)) == [master]


def test_the_cpu_offload_optimizer_steps_through_its_sub_optimizers():
    host, device = [torch.zeros(4)], [torch.zeros(2)]
    inner = types.SimpleNamespace(sub_optimizers=[optimizer(host), optimizer(device)])
    assert stepped_of(types.SimpleNamespace(optimizer=inner)) == host + device


def test_a_float32_model_is_its_own_masters():
    model = [Layer(torch.float32)]
    values = list(model[0].parameters())
    part = types.SimpleNamespace(optimizer=optimizer(values, moments(values)))
    precision = Precision(model_dtype="float32")
    evidence = precision_evidence(model, part, precision, {})
    assert evidence["masters"] == {"float32": 16}
    assert evidence["weights"] == {"float32": 16}


def test_a_pinned_float32_parameter_is_counted_apart():
    model = [Layer(torch.bfloat16, pinned=torch.float32)]
    precision = Precision(model_dtype="bfloat16", master_dtype=None)
    pinned = pinned_parameters(model, precision)
    assert pinned == {"A_log": "float32"}
    values = list(model[0].parameters())
    part = types.SimpleNamespace(optimizer=optimizer(values, moments(values)))
    evidence = precision_evidence(model, part, precision, pinned)
    assert evidence["weights"] == {"bfloat16": 16}
    assert evidence["pinned_weights"] == {"float32": 4}


def test_values_stepped_at_another_dtype_are_refused():
    model = [Layer(torch.bfloat16)]
    values = list(model[0].parameters())
    part = types.SimpleNamespace(optimizer=optimizer(values, moments(values)))
    with pytest.raises(ValueError, match="steps values at"):
        precision_evidence(model, part, Precision(model_dtype="bfloat16"), {})
