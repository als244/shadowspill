"""GLM chunk policy uses the generic compiler/provider contracts."""


def test_glm_policy_keeps_both_variants_and_local_expert_state_bounded():
    from shadowspill_workloads.policies.glm_chunks import (
        build_options,
    )

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
