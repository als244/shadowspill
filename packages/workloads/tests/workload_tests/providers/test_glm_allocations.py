"""Public GLM builds cover the Hopper calls observed in the real DP8 capture."""

import pytest

from engine_tests.training.test_allocation_coverage import compiler_fixture
from shadowspill import Parallelism
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.capture.allocation_coverage import (
    allocation_coverage,
)
from shadowspill.training.preparation.capture.templates import capture_inventory
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    LoRA,
    Memory,
    Model,
    Objective,
    Optimizer,
    Preparation,
    Weights,
)
from shadowspill_client.parallelism import data_parallel, expert_parallel

# These exact calls were missing from the full GLM Flash H100 capture. Keep this
# independent of the declaration table so a removed/renamed rule fails on CPU.
HOPPER_CALLS = (
    "torch.ops.mlops.kimi_delta_attention_cula_sm90_native_fwd.default",
    "torch.ops.mlops.kimi_delta_attention_cula_sm90_native_bwd.default",
    "torch.ops.mlops.mhc_post_tilekernels.default",
    "torch.ops.mlops.mhc_post_tilekernels_backward.default",
)


@pytest.mark.parametrize("family", ["glm53", "glm53_flash"])
@pytest.mark.parametrize("implementation", ["mlops", "mlops_ep"])
def test_public_glm_hopper_allocation_coverage(family, implementation):
    parallelism = (
        expert_parallel(8, experts=["layers.*.mlp.experts.*"])
        if implementation == "mlops_ep"
        else data_parallel(8, min_shard_bytes=0)
    )
    setup = resolve_training(
        load_registry(["shadowspill-workloads"]),
        model=Model(
            family,
            implementation=implementation,
            config={
                "preset": "tiny",
                "architecture": {
                    "hidden_size": 128,
                    "num_local_experts": 8,
                    "expert_storage": "fp8_block",
                },
            },
            lora=LoRA(16),
            initialize=Weights.random(seed=17),
        ),
        objective=Objective("cross_entropy"),
        optimizer=Optimizer("adamw"),
        memory=Memory(1, 1),
        batch=BatchConfig(
            limits={
                "token_capacity": 256,
                "max_sequence_length": 32,
                "max_sequences": 64,
            }
        ),
        parallelism=Parallelism.from_dict(parallelism),
        prepare=Preparation(candidates=BuildCandidates(microbatch_sizes=[32])),
    )
    inventory = capture_inventory(prepare_layouts(setup))
    assert inventory.templates
    for template in inventory.templates.values():
        captured = compiler_fixture(template, opaque=HOPPER_CALLS)
        proof = allocation_coverage(template, captured)
        assert set(proof["tasks"]["update_impl"]["contracts"]) == set(HOPPER_CALLS)
        # The audited exact implementations do not certify a wrapper, a new
        # provider, or a data-dependent allocation merely sharing their prefix.
        for name in HOPPER_CALLS:
            with pytest.raises(ValueError, match="missing all-value"):
                allocation_coverage(
                    template, compiler_fixture(template, opaque=[name + "_wrapper"])
                )
