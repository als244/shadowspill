from dataclasses import replace

import pytest
import torch

from workloads.full_model import build_model, initialize_model, throughput_spec
from workloads.mlops import Qwen30B, Qwen30BConfig, Qwen35B, Qwen35BConfig


def tiny(config):
    return replace(
        config,
        n_layers=4,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        n_experts=4,
        top_k=2,
        d_ff_expert=16,
        d_ff_shared=16 if config.d_ff_shared else 0,
        vocab_size=97,
        max_seq_len=16,
        lin_k_heads=2,
        lin_v_heads=4,
        lin_k_head_dim=4,
        lin_v_head_dim=4,
    )


@pytest.mark.parametrize(
    "family,count", [("qwen30b", 30_532_122_624), ("qwen35b", 34_660_610_688)]
)
def test_published_presets_construct_on_meta(family, count):
    spec = throughput_spec(family, "mlops")
    model = build_model(spec)
    assert all(p.is_meta for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == count
    assert model.embed.weight is not model.lm_head.weight
    c = model.config
    assert c.attention_width == 4096
    assert c.d_model == 2048 and c.top_k == 8 and c.norm_epsilon == 1e-6
    assert c.router_aux_loss_coef == 0.001
    if family == "qwen30b":
        assert (c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim) == (48, 32, 4, 128)
        assert (c.n_experts, c.d_ff_expert, c.d_ff_shared, c.vocab_size) == (
            128,
            768,
            0,
            151936,
        )
        assert all(block.kind == "full" for block in model.blocks)
    else:
        assert (c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim) == (40, 16, 2, 256)
        assert (c.n_experts, c.d_ff_expert, c.d_ff_shared, c.vocab_size) == (
            256,
            512,
            512,
            248320,
        )
        assert [block.kind for block in model.blocks] == [
            "linear",
            "linear",
            "linear",
            "full",
        ] * 10
        assert c.zero_centered_norm and c.attention_gate
        assert c.rotary_width == 64 and c.lin_conv_kernel == 4


@pytest.mark.parametrize(
    "cls,config", [(Qwen30B, Qwen30BConfig()), (Qwen35B, Qwen35BConfig())]
)
def test_meta_initialization_and_packed_boundaries(cls, config):
    c = tiny(config)
    with torch.device("meta"):
        model = cls(c)
    model.to_empty(device="cpu")
    initialize_model(model)
    assert torch.isfinite(model.embed.weight).all()
    assert abs(model.embed.weight.std().item() - c.initializer_range) < 0.002
    assert (model.final_norm.weight == (0 if c.zero_centered_norm else 1)).all()
    tokens = torch.randint(c.vocab_size, (1, 9))
    packed = model(tokens, (4, 5))
    independent = torch.cat((model(tokens[:, :4]), model(tokens[:, 4:])), dim=1)
    torch.testing.assert_close(packed, independent, atol=2e-6, rtol=2e-5)
    loss = model.loss(tokens, tokens, seq_lens=(4, 5))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def _reference(c):
    pytest.importorskip("transformers")
    common = dict(
        vocab_size=c.vocab_size,
        hidden_size=c.d_model,
        num_hidden_layers=c.n_layers,
        num_attention_heads=c.n_heads,
        num_key_value_heads=c.n_kv_heads,
        head_dim=c.head_dim,
        num_experts=c.n_experts,
        num_experts_per_tok=c.top_k,
        moe_intermediate_size=c.d_ff_expert,
        rms_norm_eps=c.norm_epsilon,
        max_position_embeddings=c.max_seq_len,
        tie_word_embeddings=False,
        use_cache=False,
        attention_dropout=0.0,
        router_aux_loss_coef=c.router_aux_loss_coef,
    )
    if not c.attention_gate:
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

        cfg = Qwen3MoeConfig(**common, norm_topk_prob=True, rope_theta=c.rope_base)
        cfg._attn_implementation = "eager"
        return Qwen3MoeForCausalLM(cfg)
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
    )

    cfg = Qwen3_5MoeTextConfig(
        **common,
        full_attention_interval=c.full_attention_interval,
        layer_types=[
            "full_attention" if c.layer_kind(i) == "full" else "linear_attention"
            for i in range(c.n_layers)
        ],
        linear_num_key_heads=c.lin_k_heads,
        linear_num_value_heads=c.lin_v_heads,
        linear_key_head_dim=c.lin_k_head_dim,
        linear_value_head_dim=c.lin_v_head_dim,
        linear_conv_kernel_dim=c.lin_conv_kernel,
        shared_expert_intermediate_size=c.d_ff_shared,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": c.rope_base,
            "partial_rotary_factor": c.partial_rotary_factor,
            "mrope_section": [1, 0, 0],
            "mrope_interleaved": True,
        },
    )
    cfg._attn_implementation = "eager"
    return Qwen3_5MoeForCausalLM(cfg)


def _copy_reference(model, reference):
    """Map published separate/transposed tensors to the workload's packed layout."""
    checks = []

    def bind(target, *sources, transform=lambda xs: xs[0]):
        with torch.no_grad():
            target.copy_(transform(sources))
        checks.append((target, sources, transform))

    bind(model.embed.weight, reference.model.embed_tokens.weight)
    bind(model.lm_head.weight, reference.lm_head.weight)
    bind(model.final_norm.weight, reference.model.norm.weight)
    for block, layer in zip(model.blocks, reference.model.layers, strict=True):
        bind(block.attn_norm.weight, layer.input_layernorm.weight)
        bind(block.ffn_norm.weight, layer.post_attention_layernorm.weight)
        own = block.mixer
        if block.kind == "full":
            other = layer.self_attn
            for local, remote in [
                ("wq", "q_proj"),
                ("wk", "k_proj"),
                ("wv", "v_proj"),
                ("wo", "o_proj"),
                ("q_norm", "q_norm"),
                ("k_norm", "k_norm"),
            ]:
                bind(getattr(own, local).weight, getattr(other, remote).weight)
        else:
            other = layer.linear_attn
            bind(
                own.w_qkvz.weight,
                other.in_proj_qkv.weight,
                other.in_proj_z.weight,
                transform=lambda xs: torch.cat(xs),
            )
            bind(
                own.w_ba.weight,
                other.in_proj_b.weight,
                other.in_proj_a.weight,
                transform=lambda xs: torch.cat(xs),
            )
            bind(own.w_out.weight, other.out_proj.weight)
            bind(own.conv.weight, other.conv1d.weight)
            bind(own.lin_norm.weight, other.norm.weight)
            bind(own.A_log, other.A_log)
            bind(own.dt_bias, other.dt_bias)
        bind(block.moe.router.weight, layer.mlp.gate.weight)
        bind(
            block.moe.w13,
            layer.mlp.experts.gate_up_proj,
            transform=lambda xs: xs[0].transpose(1, 2),
        )
        bind(
            block.moe.w2,
            layer.mlp.experts.down_proj,
            transform=lambda xs: xs[0].transpose(1, 2),
        )
        if block.moe.shared is not None:
            bind(
                block.moe.shared.gate_up.weight,
                layer.mlp.shared_expert.gate_proj.weight,
                layer.mlp.shared_expert.up_proj.weight,
                transform=lambda xs: torch.cat(xs),
            )
            bind(block.moe.shared.down.weight, layer.mlp.shared_expert.down_proj.weight)
            bind(block.moe.shared.gate.weight, layer.mlp.shared_expert_gate.weight)
    return checks


@pytest.mark.parametrize(
    "cls,config", [(Qwen30B, Qwen30BConfig()), (Qwen35B, Qwen35BConfig())]
)
def test_logits_and_all_parameter_gradients_match_transformers(
    cls, config, monkeypatch
):
    torch.manual_seed(607)
    c = tiny(config)
    reference = _reference(c)
    if c.attention_gate:
        import inspect

        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

        # Test the published CPU recurrence, not optional FLA/CUDA kernels.
        for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule"):
            monkeypatch.setattr(hf, name, inspect.unwrap(getattr(hf, name)))
    model = cls(c)
    checks = _copy_reference(model, reference)
    tokens = torch.randint(c.vocab_size, (1, 7))
    expected = reference(tokens, use_cache=False, output_router_logits=True)
    actual = model(tokens)
    torch.testing.assert_close(actual, expected.logits, atol=3e-6, rtol=3e-5)
    # An arbitrary upstream gradient exercises every projection without
    # conflating model math with HF's shifted-label loss convention.
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected.logits * upstream).sum().backward()
    for target, sources, transform in checks:
        gradient = transform(tuple(p.grad for p in sources))
        torch.testing.assert_close(target.grad, gradient, atol=8e-5, rtol=8e-4)
    _, auxiliary = model.hidden(tokens)
    torch.testing.assert_close(auxiliary, expected.aux_loss, atol=2e-6, rtol=2e-5)
