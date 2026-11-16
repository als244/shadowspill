"""Storage policy preserves aliases, loaded values and explicit user precedence."""

import pytest
import torch
from torch import nn

from shadowspill.pytorch.precision import apply_parameter_dtypes, parameter_dtypes


def module():
    result = nn.Module()
    result.weight = nn.Parameter(torch.tensor([0.12345679, 0.23456789]))
    result.alias = result.weight
    result.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
    result.register_buffer("offset", torch.tensor([0.34567891]))
    result.register_buffer("offset_alias", result.offset)
    result.register_buffer("indices", torch.arange(2))
    return result


def test_defaults_preserve_original_values_aliases_gradients_and_small_updates():
    model = module()
    identities = {name: id(tensor) for name, tensor in model.named_parameters()}
    original = model.weight.detach().clone()
    gradient = torch.tensor([1 / 128, 1 / 64])
    model.weight.grad = gradient.clone()
    apply_parameter_dtypes(
        model, defaults={"*": "bfloat16", "weight": "float32", "optional.*": "float32"}
    )
    assert identities == {name: id(tensor) for name, tensor in model.named_parameters()}
    assert model.weight is model.alias and model.offset is model.offset_alias
    assert not model.frozen.requires_grad
    assert model.frozen.dtype == model.offset.dtype == torch.bfloat16
    assert model.indices.dtype == torch.int64
    assert model.weight.dtype == model.weight.grad.dtype == torch.float32
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.grad, gradient, rtol=0, atol=0)
    torch.optim.SGD(model.parameters(), lr=1 / 1024, foreach=False).step()
    assert not torch.equal(model.weight, original)
    torch.testing.assert_close(
        model.weight,
        (original.double() - gradient.double() / 1024).float(),
        rtol=0,
        atol=0,
    )


def test_overrides_win_over_more_specific_defaults_and_cast_existing_gradients():
    model = module()
    model.weight.grad = torch.ones_like(model.weight)
    apply_parameter_dtypes(
        model,
        defaults={"weight": "float32", "offset": "float64"},
        overrides={"*": "bfloat16", "offset_alias": "float32"},
    )
    assert model.weight.dtype == model.weight.grad.dtype == torch.bfloat16
    assert model.weight is model.alias
    assert model.offset.dtype == torch.float32
    assert model.offset is model.offset_alias


@pytest.mark.parametrize(
    "rules, message",
    [
        ({"missing.*": "float32"}, "matches no registered tensor"),
        ({"indices": "float32"}, "matches no registered tensor"),
        ({"weight": "float32", "alias": "bfloat16"}, "conflicting declarations"),
        ({"w*": "float32", "*t": "bfloat16"}, "conflicting declarations"),
    ],
)
def test_invalid_selectors_do_not_partially_change_the_model(rules, message):
    model = module()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(ValueError, match=message):
        apply_parameter_dtypes(model, overrides={"frozen": "float16", **rules})
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "rules",
    [
        None,
        [],
        {"": "float32"},
        {" weight": "float32"},
        {3: "float32"},
        {"weight": "fp32"},
        {"weight": "int32"},
        {"weight": "float8_e4m3fn"},
        {"weight": 32},
    ],
)
def test_portable_rule_validation(rules):
    with pytest.raises(
        (TypeError, ValueError), match=r"parameter_dtype|parameter dtype"
    ):
        parameter_dtypes(rules)


def test_tensor_subclass_conversion_requires_its_provider():
    class Storage(torch.Tensor):
        pass

    model = nn.Module()
    model.register_buffer("custom", torch.ones(2).as_subclass(Storage))
    with pytest.raises(TypeError, match="installed provider"):
        apply_parameter_dtypes(model, overrides={"custom": "bfloat16"})
    assert model.custom.dtype == torch.float32
