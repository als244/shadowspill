"""Numerical and ownership checks for the isolated design probe."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from lora_prototype import ExpertLoRA, LoRAConfig, LoRALinear, apply_lora


def test_dense_zero_update_preserves_base_and_input_gradient():
    torch.manual_seed(17)
    base = nn.Linear(7, 11, dtype=torch.float64).eval()
    original = copy.deepcopy(base)
    weight = base.weight
    layer = apply_lora(base, LoRAConfig(rank=3, alpha=5), targets=[''])
    assert layer.base.weight is weight
    assert not layer.training
    x = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    y = layer(x)
    torch.testing.assert_close(y, original(x), rtol=0, atol=0)
    dx = torch.autograd.grad(y.sum(), x)[0]
    expected = torch.autograd.grad(original(x).sum(), x)[0]
    torch.testing.assert_close(dx, expected, rtol=0, atol=0)
    assert not weight.requires_grad


@pytest.mark.parametrize('compiled', [False, True])
def test_dense_nonzero_factor_gradients(compiled):
    torch.manual_seed(21)
    base = nn.Linear(7, 11, dtype=torch.float64)
    layer = apply_lora(base, LoRAConfig(rank=3, alpha=5), targets=[''])
    nn.init.normal_(layer.lora_b)
    reference = copy.deepcopy(layer)
    x = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    call = torch.compile(layer, backend='aot_eager', fullgraph=True) if compiled else layer
    actual = call(x)
    merged = reference.base.weight + reference.lora_config.scale * (reference.lora_b @ reference.lora_a)
    expected = F.linear(ref_x, merged, reference.base.bias)
    torch.testing.assert_close(actual, expected)
    seed = torch.randn_like(actual)
    (actual * seed).sum().backward()
    (expected * seed).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for name in ('lora_a', 'lora_b'):
        torch.testing.assert_close(getattr(layer, name).grad, getattr(reference, name).grad)
    assert layer.base.weight.grad is None
    assert layer.base.bias.grad is None


def test_updates_allocate_only_factor_optimizer_state():
    model = nn.Sequential(nn.Linear(7, 11), nn.SiLU(), nn.Linear(11, 3))
    originals = {id(p): p.detach().clone() for p in model.parameters()}
    apply_lora(model, LoRAConfig(rank=2), targets=['0', '2'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(5, 7)).square().mean().backward()
        optimizer.step()
    for p in model.parameters():
        if id(p) in originals:
            torch.testing.assert_close(p, originals[id(p)], rtol=0, atol=0)
            assert p.grad is None and p not in optimizer.state
        else:
            assert p in optimizer.state


def test_aliases_and_full_training_selection_preserve_identity():
    model = nn.Module()
    model.projection = nn.Linear(5, 5)
    model.alias = model.projection
    model.embed = nn.Embedding(9, 5)
    model.head = nn.Linear(5, 9, bias=False)
    model.head.weight = model.embed.weight
    weight = model.projection.weight
    apply_lora(model, LoRAConfig(rank=2), targets=['projection'], trainable_base=['head.weight'])
    assert model.projection is model.alias
    assert model.projection.base.weight is weight
    assert model.embed.weight is model.head.weight
    assert model.head.weight.requires_grad
    assert not weight.requires_grad
    assert len(list(model.parameters())) == 5


@pytest.mark.parametrize('targets,trainable', [(['absent'], []), (['0','missing'], []), (['0'], ['absent'])])
def test_invalid_selection_does_not_mutate_model(targets, trainable):
    model = nn.Sequential(nn.Linear(2, 2))
    original = model[0]
    with pytest.raises(ValueError):
        apply_lora(model, LoRAConfig(rank=1), targets=targets, trainable_base=trainable)
    assert model[0] is original
    assert all(p.requires_grad for p in model.parameters())


def test_meta_materialization_and_custom_base_initializer():
    class ConstantLinear(nn.Linear):
        def reset_parameters(self):
            nn.init.constant_(self.weight, 0.125)
            nn.init.constant_(self.bias, -0.5)

    with torch.device('meta'):
        model = nn.Sequential(ConstantLinear(7, 11))
        apply_lora(model, LoRAConfig(rank=3), targets=['0'])
    model.to_empty(device='cpu')
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    torch.testing.assert_close(model[0].base.weight, torch.full((11, 7), 0.125))
    torch.testing.assert_close(model[0].lora_b, torch.zeros((11, 3)), rtol=0, atol=0)
    assert not model[0].base.weight.requires_grad
    assert model[0].lora_a.requires_grad
    checkpoint = model.state_dict()
    restored = copy.deepcopy(model)
    restored.load_state_dict(checkpoint)
    torch.testing.assert_close(model(torch.ones(2, 7)), restored(torch.ones(2, 7)), rtol=0, atol=0)


@pytest.mark.parametrize('nonzero', [False, True])
def test_expert_factors_and_input_gradients_match_merged_oracle(nonzero):
    torch.manual_seed(8)
    experts, width, hidden = 4, 7, 5
    gate_up = nn.Parameter(torch.randn(experts, width, 2 * hidden, dtype=torch.float64))
    down = nn.Parameter(torch.randn(experts, hidden, width, dtype=torch.float64))
    layer = ExpertLoRA(gate_up, down, LoRAConfig(rank=3, alpha=5))
    if nonzero:
        nn.init.normal_(layer.lora_gate_up_b)
        nn.init.normal_(layer.lora_down_b)
    ref = copy.deepcopy(layer)
    # Expert 3 deliberately receives no tokens.
    ids = torch.tensor([[0, 1], [1, 2], [2, 0]])
    weights = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    ref_weights = weights.detach().clone().requires_grad_(True)
    x = torch.randn(3, width, dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    actual = layer(x, ids, weights)
    w13 = ref.gate_up + ref.lora_config.scale * (ref.lora_gate_up_a @ ref.lora_gate_up_b)
    w2 = ref.down + ref.lora_config.scale * (ref.lora_down_a @ ref.lora_down_b)
    expected = torch.zeros_like(x)
    for expert in range(experts):
        gate, up = (ref_x @ w13[expert]).chunk(2, -1)
        y = (F.silu(gate) * up) @ w2[expert]
        expected = expected + (ref_weights * (ids == expert)).sum(-1)[:, None] * y
    torch.testing.assert_close(actual, expected)
    seed = torch.randn_like(actual)
    (actual * seed).sum().backward()
    (expected * seed).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(weights.grad, ref_weights.grad)
    for name, p in layer.named_parameters():
        if name.startswith('lora_'):
            torch.testing.assert_close(p.grad, dict(ref.named_parameters())[name].grad)
            torch.testing.assert_close(p.grad[3], torch.zeros_like(p.grad[3]), rtol=0, atol=0)
        else:
            assert p.grad is None
