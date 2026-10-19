"""Exercise the normal checkpoint's names and exact stacked FP8 import."""

import pytest
import torch

pytest.importorskip("mlops")
pytest.importorskip("safetensors.torch")

from tests.workloads.test_glm53_checkpoint import _materialized, _write
from workloads.mlops.glm53 import Config, LanguageModel
from workloads.mlops.glm53.checkpoint import GLMCheckpoint, source_name
from workloads.mlops.glm53_flash.checkpoint import physical_state


@pytest.mark.parametrize("use_writer", [False, True])
def test_normal_checkpoint_streams_all_layers_and_preserves_fp8_bytes(
    tmp_path, use_writer
):
    from shadowspill.pytorch import initialize_storage_

    config = Config.tiny()
    experts, original = _write(
        tmp_path,
        fp8_experts=True,
        config=config,
        model_type=LanguageModel,
        name_for=source_name,
        prefix="model",
        nested=False,
    )
    source = GLMCheckpoint(tmp_path, expert_storage="fp8_block")
    try:
        model = _materialized(
            source.build_model(
                lora_rank=16,
                lora_factor_dtype=torch.bfloat16,
            )
        )
        source.load_into(
            model, write_tensor=initialize_storage_ if use_writer else None
        )
        assert source.inventory["converted_expert_banks"] == 0
        assert model.config.indexer_types == config.indexer_types
        assert model.layers[1].self_attn.indexer is None
        for index, (w13, w2) in experts.items():
            for weight, expected in (
                (model.layers[index].mlp.w13, w13),
                (model.layers[index].mlp.w2, w2),
            ):
                assert torch.equal(weight.dequantize(dtype=torch.float32), expected)
        actual = physical_state(model)
        for name, value in physical_state(original).items():
            if name.endswith((".mlp.w13", ".mlp.w2")):
                continue
            torch.testing.assert_close(actual[name], value)
        for name, value in model.named_parameters():
            if value.requires_grad:
                assert value.dtype == torch.bfloat16, name
    finally:
        source.close()
