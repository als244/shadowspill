"""Exact EP allocation scope and provider metadata, before distributed profiling."""

import pytest
import torch

from engine_tests.training.test_allocation_coverage import compiler_fixture
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.capture.allocation_coverage import (
    allocation_coverage,
)
from shadowspill_workloads.definitions import expert_allocations, source
from workload_tests.definitions.test_expert_definitions import FAMILIES, prepared


@pytest.mark.parametrize("family", FAMILIES)
def test_only_audited_bf16_operators_are_covered(family):
    _, inventory = prepared(family)
    template = next(iter(inventory.templates.values()))
    calls = tuple(expert_allocations.CONTRACTS)
    proof = allocation_coverage(template, compiler_fixture(template, opaque=calls))
    assert proof["tasks"]["update_impl"]["contracts"] == expert_allocations.CONTRACTS
    assert proof["preparations"] == {}
    for unsupported in (
        "torch.ops.mlops_ep_quack_v1.fp8_forward.default",
        "torch.ops.mlops_ep_quack_v1.fp8_backward.default",
        "torch.ops.mlops_ep_quack_shared_v1.forward.default",
        "torch.ops.mlops_ep_quack_output_only_v1.forward.default",
        "torch.ops.mlops_ep_quack_lora.forward.default",
    ):
        with pytest.raises(ValueError, match="missing all-value"):
            allocation_coverage(
                template, compiler_fixture(template, opaque=[unsupported])
            )


@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_saved_plan_backing_storage_covers_moonep_vector_writes(grad_dtype):
    from mlops.expert_parallel import QuackMoEConfig
    from mlops.expert_parallel.quack.config import config_signature
    from mlops.expert_parallel.quack.operators import _b_fake, _f_fake

    # Odd S and q exercise the four-int32 backing-store rounding. This is fake
    # metadata verification, not an assertion that the native kernels ran.
    tokens, experts, ranks, top_k, width, hidden = 5, 6, 2, 2, 128, 128
    local = experts // ranks
    capacity = tokens * top_k + 2 * local * 127
    config = QuackMoEConfig(
        ep_size=ranks,
        num_experts=experts,
        top_k=top_k,
        model_dim=width,
        expert_hidden_dim=hidden,
        tokens_per_rank=tokens,
        weight_grad_dtype=grad_dtype,
        share_expert_banks=True,
        gemm_tuned=False,
    )
    x = torch.empty(tokens, width, dtype=torch.bfloat16, device="meta")
    weights = [x.new_empty(local, 2 * hidden, width), x.new_empty(local, width, hidden)]
    output, state = _f_fake(x, None, None, weights, None, 1, config_signature(config))
    assert output.shape == x.shape and output.dtype == x.dtype
    expected = (
        (tokens * top_k,),
        (ranks, local),
        (experts + local, 2),
        (2,),
        (capacity, 3),
        (capacity,),
        (2,),
    )
    assert len(state) == 11
    for tensor, shape in zip(state[:7], expected, strict=True):
        assert tuple(tensor.shape) == shape and tensor.dtype == torch.int32
        assert tensor.untyped_storage().nbytes() == ((tensor.numel() + 3) // 4 * 4) * 4
    assert [tuple(value.shape) for value in state[7:]] == [
        (2 * local + 1,),
        (capacity, width),
        (capacity,),
        (capacity, 2 * hidden),
    ]
    dx, dp, gradients = _b_fake(x, weights, state, 1, config_signature(config))
    assert dx.shape == x.shape and dx.dtype == x.dtype
    assert dp.shape == (tokens, top_k) and dp.dtype == torch.float32
    assert [g.shape for g in gradients] == [w.shape for w in weights]
    assert all(g.dtype == grad_dtype for g in gradients)


def test_moonep_changes_invalidate_all_supplied_moe_definitions(monkeypatch):
    changed = False

    def digest(name):
        return (name + ("changed" if changed and name == "moonep" else "")).encode()

    monkeypatch.setattr(source, "_provider_digest", digest)

    def identities():
        registry = load_registry(["shadowspill-workloads"])
        return {
            family: registry.resolve_model(f"shadowspill/{family}@1").source
            for family in FAMILIES
        }

    before = identities()
    changed = True
    after = identities()
    assert all(before[family] != after[family] for family in FAMILIES)
