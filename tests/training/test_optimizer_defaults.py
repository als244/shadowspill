"""Training policies do not change standalone optimizers or numerical references."""

from functools import partial

import pytest
import torch
from torch import nn

from shadowspill.training._model import OptimizerFactory
from workloads.numerical import build_case
from workloads.precision import TrainingDtypes
from workloads.recipes.text.quickstart import Precision


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_training_defaults_follow_moment_dtype_and_leave_mlops_unchanged(dtype):
    from mlops.optim import AdamW

    model = nn.Linear(3, 2, device="meta", dtype=torch.float32)
    factory = OptimizerFactory.bind(model, AdamW, {"opt_state_dtype": dtype})
    optimizer = factory(model.parameters())
    expected = "stochastic" if dtype == torch.bfloat16 else "nearest"
    assert optimizer.param_groups[0]["opt_state_rounding"] == expected
    standalone = AdamW(model.parameters(), opt_state_dtype=dtype)
    assert standalone.param_groups[0]["opt_state_rounding"] == "nearest"


@pytest.mark.parametrize("source", ["arguments", "partial", "groups"])
def test_explicit_rounding_overrides_training_default(source):
    from mlops.optim import AdamW

    model = nn.Linear(3, 2, device="meta")
    arguments = {"opt_state_rounding": "nearest"} if source == "arguments" else {}
    constructor = (
        partial(AdamW, opt_state_rounding="nearest") if source == "partial" else AdamW
    )

    def groups(model):
        return [{"params": model.parameters(), "opt_state_rounding": "nearest"}]

    factory = OptimizerFactory.bind(
        model, constructor, arguments, groups if source == "groups" else None
    )
    optimizer = factory(model.parameters())
    assert optimizer.param_groups[0]["opt_state_rounding"] == "nearest"


def test_parameter_dtype_policy_preserves_order_names_and_group_options():
    from mlops.optim import AdamW

    model = nn.ParameterDict(
        {
            "a": nn.Parameter(torch.empty(2, device="meta", dtype=torch.bfloat16)),
            "b": nn.Parameter(torch.empty(2, device="meta", dtype=torch.float16)),
            "c": nn.Parameter(torch.empty(2, device="meta", dtype=torch.bfloat16)),
        }
    )

    def groups(model):
        return [
            {
                "params": model.parameters(),
                "param_names": ["a", "b", "c"],
                "weight_decay": 0.2,
            }
        ]

    factory = OptimizerFactory.bind(
        model, AdamW, {"opt_state_dtype": "parameter"}, groups
    )
    optimizer = factory(model.parameters())
    assert [g["opt_state_rounding"] for g in optimizer.param_groups] == [
        "stochastic",
        "nearest",
        "stochastic",
    ]
    assert [g["param_names"] for g in optimizer.param_groups] == [["a"], ["b"], ["c"]]
    assert all(g["weight_decay"] == 0.2 for g in optimizer.param_groups)
    assert [id(p) for g in optimizer.param_groups for p in g["params"]] == [
        id(p) for p in model.parameters()
    ]


def test_group_dtypes_and_rounding_override_constructor_defaults():
    from mlops.optim import AdamW

    model = nn.Linear(3, 2, device="meta")

    def groups(model):
        return [
            {"params": [model.weight], "opt_state_dtype": torch.bfloat16},
            {"params": [model.bias], "opt_state_dtype": torch.float16},
        ]

    constructor = partial(AdamW, opt_state_dtype=torch.float32)
    optimizer = OptimizerFactory.bind(model, constructor, {}, groups)(
        model.parameters()
    )
    assert [g["opt_state_rounding"] for g in optimizer.param_groups] == [
        "stochastic",
        "nearest",
    ]


def test_quickstart_and_performance_default_to_stochastic_but_numerical_does_not():
    parameters = [nn.Parameter(torch.empty(2, device="meta"))]
    factory = TrainingDtypes().optimizer()
    assert factory(parameters).param_groups[0]["opt_state_rounding"] == "stochastic"
    quickstart = Precision(opt_state_rounding="nearest").optimizer(factory)
    assert quickstart(parameters).param_groups[0]["opt_state_rounding"] == "nearest"
    numerical = build_case("llama3").optimizer(parameters)
    assert numerical.param_groups[0]["opt_state_rounding"] == "nearest"


def test_torch_adamw_keeps_its_own_options():
    model = nn.Linear(3, 2)
    optimizer = OptimizerFactory.bind(model, torch.optim.AdamW, {})(model.parameters())
    assert "opt_state_rounding" not in optimizer.param_groups[0]
