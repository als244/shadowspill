"""GLM composition keeps required FP32 arithmetic under caller autocast."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytest.importorskip("mlops")
from workloads.models.shared.glm.common import without_autocast


@pytest.mark.parametrize("enabled", (False, True))
def test_fp32_projection_preserves_precision_and_restores_autocast(enabled):
    class Projection(nn.Module):
        def forward(self, x, weight):
            with without_autocast(x):
                return F.linear(x.float(), weight.float())

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7
    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 11
    reference = F.linear(x, weight)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
        model = Projection()
        exported = torch.export.export(model, (x, weight), strict=True).module()
        for call in (model, exported):
            actual = call(x, weight)
            assert actual.dtype is torch.float32
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert torch.is_autocast_enabled("cpu") is enabled


@pytest.mark.parametrize("precision", ("fp16", "fp8", "nvfp4", "mxfp8"))
def test_unimplemented_checkpoint_compute_fails_before_reading_files(
    tmp_path, precision
):
    from workloads.models.glm53_flash.checkpoint import GLMCheckpoint

    with pytest.raises(ValueError, match="BF16 GEMMs only"):
        GLMCheckpoint(tmp_path / "not-downloaded", gemm_precision=precision)


@pytest.mark.parametrize("head", (False, True))
def test_multimodal_lora_freezes_base_and_selects_only_projection_factors(head):
    from workloads.models.glm53_flash.config import Config
    from workloads.models.glm53_flash.implementations.mlops import Model, VisionConfig

    config = Config.tiny(hidden_size=128, lora_rank=8, lora_alpha=8, lora_head=head)
    vision = VisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        out_hidden_size=128,
        projection_intermediate_size=128,
        patch_size=2,
    )
    model = Model(config, device="meta", vision_config=vision)
    trainable = {
        name for name, value in model.named_parameters() if value.requires_grad
    }
    assert "visual.blocks.0.attn.qkv.lora_a" in trainable
    assert "layers.0.self_attn.q_proj.lora_b" in trainable
    gates = (
        "forget_gate.f_a_proj",
        "forget_gate.f_b_proj",
        "b_proj",
        "g_a_proj",
        "g_b_proj",
    )
    for layer in model.layers:
        if hasattr(layer.self_attn, "forget_gate"):
            for name in gates:
                module = layer.self_attn.get_submodule(name)
                assert isinstance(module, nn.Linear)
                assert not module.weight.requires_grad
                assert not hasattr(module, "lora_config")
    assert "layers.0.mlp.gate_proj.lora_b" in trainable
    assert "layers.1.mlp.lora_gate_up_a" in trainable
    assert not model.layers[1].mlp.w13.requires_grad
    assert ("lm_head.lora_a" in trainable) is head
    assert all(
        name.endswith(
            ("lora_a", "lora_b", "gate_up_a", "gate_up_b", "down_a", "down_b")
        )
        for name in trainable
    )
    assert not any(
        ".indexer." in name or ".shared_experts." in name for name in trainable
    )
    assert not model.visual.patch_embed.proj.weight.requires_grad
    assert not model.visual.downsample.weight.requires_grad


@pytest.mark.parametrize("precision", ["fp32", "activation"])
def test_flash_decoder_uses_mhc_semantic_interfaces(monkeypatch, precision):
    from workloads.models.glm53_flash.config import Config
    from workloads.models.glm53_flash.implementations.mlops import model as model_module
    from workloads.models.shared.glm import common

    config = Config.tiny(hc_coefficient_precision=precision)
    layer = model_module.DecoderLayer(config, 0, "cpu")

    class Branch(nn.Module):
        def forward(self, x, *metadata):
            return x * 0.25

    layer.self_attn = Branch()
    layer.mlp = Branch()
    calls = []
    pre, combine = common.hc.mhc_pre, model_module.mhc_combine

    def observed_pre(*args, **kwargs):
        calls.append("pre")
        return pre(*args, **kwargs)

    def observed_combine(*args, **kwargs):
        calls.append(("combine", kwargs["coefficient_precision"]))
        return combine(*args, **kwargs)

    monkeypatch.setattr(common.hc, "mhc_pre", observed_pre)
    monkeypatch.setattr(model_module, "mhc_combine", observed_combine)
    x = torch.randn(3, 4, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    result = layer(x, None, None, None)
    torch.autograd.grad(result.float().square().mean(), x)
    assert calls == ["pre", ("combine", precision), "pre", ("combine", precision)]
    assert result.shape == x.shape and result.dtype == x.dtype
