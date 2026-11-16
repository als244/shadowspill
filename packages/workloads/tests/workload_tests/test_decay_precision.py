"""Dense and MoE decay storage follows the shared setup policy."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from shadowspill.pytorch.precision import apply_parameter_dtypes
from shadowspill_workloads.models.qwen35.config import PRESETS as DENSE
from shadowspill_workloads.models.qwen35.implementations.pytorch import linear_attention
from shadowspill_workloads.models.qwen35moe.config import PRESETS as MOE
from shadowspill_workloads.models.shared.delta_net import GatedDeltaNet as DenseMLOps
from shadowspill_workloads.models.shared.qwen_moe.attention import GatedDeltaNet as MoE


@pytest.fixture(params=["dense_pytorch", "dense_mlops", "moe_mlops"])
def factory(request):
    cls, preset = {
        "dense_pytorch": (linear_attention.GatedDeltaNet, DENSE["tiny"]),
        "dense_mlops": (DenseMLOps, DENSE["tiny"]),
        "moe_mlops": (MoE, MOE["tiny"]),
    }[request.param]
    config = replace(
        preset,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        lin_k_heads=2,
        lin_v_heads=4,
        lin_k_head_dim=4,
        lin_v_head_dim=4,
    )
    return lambda: cls(config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_meta_construction_and_materialization_preserve_decay_precision(factory, dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("meta"):
            module = factory()
    finally:
        torch.set_default_dtype(previous)
    assert module.A_log.dtype == module.dt_bias.dtype == torch.float32
    assert module.w_qkvz.weight.dtype == dtype
    module.to_empty(device="cpu")
    module.reset_parameters()
    assert module.A_log.dtype == module.dt_bias.dtype == torch.float32
    assert module.w_qkvz.weight.dtype == dtype
    assert torch.isfinite(module.A_log).all() and torch.isfinite(module.dt_bias).all()


def test_shared_policy_preserves_loaded_decay_values_gradients_and_small_updates(
    factory,
):
    module = factory()
    original = {
        "A_log": torch.tensor([0.12345679, 0.23456789, 0.34567891, 0.45678912]),
        "dt_bias": torch.tensor([-0.56789123, -0.67891234, -0.78912345, -0.89123456]),
    }
    module.load_state_dict({**module.state_dict(), **original})
    gradients = torch.arange(1, 5, dtype=torch.float32) / 128
    for name in original:
        getattr(module, name).grad = gradients.clone()

    # The policy converts each original tensor directly to its selected dtype.
    parent = nn.Sequential(module)
    for dtype in ("bfloat16", "float16", "float64", "bfloat16"):
        apply_parameter_dtypes(
            parent,
            defaults={
                "*": dtype,
                "0.A_log": "float32",
                "0.dt_bias": "float32",
            },
        )
        assert module.w_qkvz.weight.dtype == getattr(torch, dtype)
        for name, expected in original.items():
            value = getattr(module, name)
            assert value.dtype == value.grad.dtype == torch.float32
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, gradients, rtol=0, atol=0)

    optimizer = torch.optim.SGD(module.parameters(), lr=1 / 1024, foreach=False)
    optimizer.step()
    for name, initial in original.items():
        value = getattr(module, name)
        expected = (initial.double() - gradients.double() / 1024).float()
        assert value.dtype == torch.float32
        assert not torch.equal(value, initial)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("factory", ["dense_mlops", "moe_mlops"], indirect=True)
def test_bfloat16_gpu_recurrence_matches_reference_decay_gradients(factory):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import mlops

    torch.manual_seed(531)
    module = factory()
    reference = linear_attention.GatedDeltaNet(module.config)
    for model in (module, reference):
        apply_parameter_dtypes(
            model,
            defaults={
                "*": "bfloat16",
                "A_log": "float32",
                "dt_bias": "float32",
            },
        )
        model.to(device="cuda")
    reference.load_state_dict(module.state_dict())
    hidden = torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16)
    tangent = torch.randn_like(hidden)
    lengths = (3, 5)
    cumulative, chunks = mlops.prepare_packed_sequence_metadata(
        lengths, torch.zeros(1, 8, device="cuda", dtype=torch.int64)
    )
    actual = module(hidden, cumulative, chunks)
    expected = reference(hidden, lengths)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=1e-5)
    (actual.float() * tangent.float()).sum().backward()
    (expected.float() * tangent.float()).sum().backward()
    for name in ("A_log", "dt_bias"):
        value = getattr(module, name)
        gradient = getattr(reference, name).grad
        assert value.dtype == value.grad.dtype == torch.float32
        assert torch.count_nonzero(gradient) > 0
        torch.testing.assert_close(value.grad, gradient, rtol=0.02, atol=1e-7)
