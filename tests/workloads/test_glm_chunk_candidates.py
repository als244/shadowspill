"""The recipe crosses explicit execution/retention controls without CUDA."""

import pytest

from workloads.recipes.text.glm53_flash_lora.settings import build_variants
from workloads.recipes.text.phases.candidates import build_candidates, program_paths


def test_candidate_product_deduplicates_unchunked_and_preserves_batch_paths():
    options = dict(
        sequence_length=32768,
        sequences_per_step=8,
        microbatch_sizes=[2, 4],
        chunk_sizes=[0, 4096, 8192],
        moe_chunk_sizes=[0, 32768],
        graph_pair_policies=["cost-input", "cost-attention", "attention-input"],
    )
    assert program_paths(options) == (
        "batch-2/training.ssp",
        "batch-2/evaluation.ssp",
        "batch-4/training.ssp",
        "batch-4/evaluation.ssp",
    )
    options["build_variants"] = build_variants(options)
    assert len(options["build_variants"]) == 15
    candidates = build_candidates(options)
    assert len(candidates) == 30
    assert {c["sequences_per_example"] for c in candidates} == {2, 4}
    assert len(set(program_paths(options))) == 60
    for policy in options["graph_pair_policies"]:
        assert {
            (c["chunk_size"], c["moe_chunk_size"])
            for c in candidates
            if c["graph_pair_policy"] == policy
        } == {
            (0, 0),
            (4096, 4096),
            (4096, 32768),
            (8192, 8192),
            (8192, 32768),
        }


@pytest.mark.parametrize(
    "sizes,experts,sequence",
    [([48], [0], 32768), ([4096], [0], 32769), ([8192], [4096], 32768)],
)
def test_invalid_fused_geometry_is_refused_before_build(sizes, experts, sequence):
    with pytest.raises(ValueError):
        build_variants(
            dict(chunk_sizes=sizes, moe_chunk_sizes=experts, sequence_length=sequence)
        )


def test_glm_policy_keeps_both_variants_and_local_expert_state_bounded():
    from workloads.recipes.text.glm53_flash_lora.policies import build_options

    for name in ("cost-input", "cost-attention", "attention-input"):
        policy = build_options(
            dict(
                sequence_length=32768,
                chunk_size=4096,
                moe_chunk_size=32768,
                graph_pair_policy=name,
            ),
            1,
        )
        chunks = policy["chunking"]
        assert chunks.variants == chunks.chunking.variants == ("save", "recompute")
        pair = policy["graph_pair_policy"]
        rules = {r.target: r.action for r in pair.save.rules}
        assert rules["mlops.sonicmoe_routing_metadata.default"] == "recompute"
        assert rules["mlops.lora_sonicmoe_up_fwd.default"] == "recompute"
        assert "mlops.lora_route.default" not in rules
        assert pair.save.default == (
            "recompute" if name == "attention-input" else "cost"
        )
        assert pair.recompute.default == "recompute"
