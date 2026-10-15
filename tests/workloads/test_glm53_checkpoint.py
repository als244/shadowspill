"""GLM checkpoint import into stacked expert banks, on a synthetic checkpoint.

A small random GLM is written under the HF names the importer reads -- the
routed experts one projection per expert, dense or as FP8 with 128x128 block
scales -- and imported into each expert storage. Everything runs on the CPU
(conversions use the GPU when there is one).
"""

import json
from dataclasses import fields

import pytest
import torch
from torch import nn

pytest.importorskip("mlops")
safetensors_torch = pytest.importorskip("safetensors.torch")

from mlops.kernels.block_fp8 import dequantize_blocks, quantize_blocks  # noqa: E402

from workloads.mlops.glm53_flash import Config, LanguageModel  # noqa: E402
from workloads.mlops.glm53_flash.checkpoint import (  # noqa: E402
    GLMCheckpoint,
    physical_state,
    source_name,
)

# Widths that are whole 128x128 blocks, as an FP8 checkpoint's experts are.
CONFIG = Config.tiny(hidden_size=128, moe_intermediate_size=128)


# Run options of this package, not fields of the published config.
OPTIONS = {"expert_storage", "gemm_precision", "lora_rank", "lora_alpha", "lora_head"}


def _text_config(config):
    values = {}
    for field in fields(Config):
        value = getattr(config, field.name)
        if field.name not in OPTIONS and isinstance(value, (bool, int, float, str)):
            values[field.name] = value
    values["n_routed_experts"] = config.num_local_experts
    return values


def _write(directory, *, fp8_experts):
    """The checkpoint of a random model; returns its expert weights per layer
    as ``{layer: (w13 [E, 2I, H], w2 [E, H, I])}`` in fp32 as stored (FP8
    experts dequantized), and the model."""
    torch.manual_seed(0)
    model = LanguageModel(CONFIG, device="cpu")
    tensors, experts = {}, {}
    for name, value in physical_state(model).items():
        value = value.detach()
        if name.endswith((".mlp.w13", ".mlp.w2")):
            continue
        if name.endswith(".self_attn.conv1d.weight"):
            prefix = source_name(name).removesuffix("conv1d.weight")
            for letter, part in zip("qkv", value.chunk(3), strict=True):
                tensors[prefix + letter + "_conv1d.weight"] = part.contiguous()
            continue
        tensors[source_name(name)] = value.contiguous()
    width = CONFIG.moe_intermediate_size
    for index, layer in enumerate(model.layers):
        if not hasattr(layer.mlp, "w13"):
            continue
        prefix = f"model.language_model.layers.{index}.mlp.experts"
        stored = []
        for weight in (layer.mlp.w13.detach(), layer.mlp.w2.detach()):
            if fp8_experts:
                data, scale = quantize_blocks(weight.float())
                stored.append((data, scale, dequantize_blocks(data, scale)))
            else:
                stored.append((weight, None, weight.float()))
        (gate_up, gate_up_scale, w13), (down, down_scale, w2) = stored
        experts[index] = (w13, w2)
        for expert in range(CONFIG.num_local_experts):
            projections = {
                "gate_proj": (gate_up[expert, :width], gate_up_scale, 0),
                "up_proj": (gate_up[expert, width:], gate_up_scale, 1),
                "down_proj": (down[expert], down_scale, None),
            }
            for projection, (weight, scale, half) in projections.items():
                key = f"{prefix}.{expert}.{projection}"
                tensors[key + ".weight"] = weight.contiguous()
                if scale is not None:
                    blocks = scale[expert]
                    if half is not None:
                        blocks = blocks.chunk(2)[half]
                    tensors[key + ".weight_scale_inv"] = blocks.contiguous()
    safetensors_torch.save_file(tensors, str(directory / "model.safetensors"))
    text = _text_config(CONFIG)
    if fp8_experts:
        text["quantization_config"] = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        }
    (directory / "config.json").write_text(json.dumps({"text_config": text}))
    return experts, model


def _materialized(model):
    """The meta model with empty CPU state, FP8 containers kept as such."""
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            flatten = getattr(parameter, "__tensor_flatten__", None)
            if flatten is None:
                value = torch.empty_like(parameter, device="cpu")
            else:
                names, metadata = flatten()
                value = type(parameter).__tensor_unflatten__(
                    {
                        n: torch.empty_like(getattr(parameter, n), device="cpu")
                        for n in names
                    },
                    metadata,
                    None,
                    None,
                )
            setattr(
                module, name, nn.Parameter(value, requires_grad=parameter.requires_grad)
            )
        for name, buffer in list(module.named_buffers(recurse=False)):
            module._buffers[name] = torch.empty_like(buffer, device="cpu")
    return model


def _load(directory, **options):
    with GLMCheckpoint(
        directory, expert_storage=options.pop("storage", None)
    ) as source:
        model = _materialized(source.build_model(**options))
        source.load_into(model)
        return model, source.inventory


def _relative(actual, expected):
    actual, expected = actual.float(), expected.float()
    return float((actual - expected).norm() / expected.norm())


def test_a_dense_checkpoint_loads_exactly_into_bf16_banks(tmp_path):
    experts, original = _write(tmp_path, fp8_experts=False)
    model, inventory = _load(tmp_path)
    assert inventory["expert_storage"] == "bf16"
    assert inventory["checkpoint_expert_format"] == "bf16"
    assert inventory["converted_expert_banks"] == 0
    expected = physical_state(original)
    for name, value in physical_state(model).items():
        assert torch.equal(value, expected[name]), name
    for index, (w13, w2) in experts.items():
        mlp = model.layers[index].mlp
        assert torch.equal(mlp.w13.float(), w13) and torch.equal(mlp.w2.float(), w2)


def test_an_fp8_checkpoint_keeps_its_blocks_byte_for_byte_by_default(tmp_path):
    from mlops.block_fp8 import BlockFP8Weight

    experts, _ = _write(tmp_path, fp8_experts=True)
    # One copy by default: no second, transposed copy of a compressed base.
    model, inventory = _load(tmp_path)
    assert inventory["expert_storage"] == "fp8_block"
    assert inventory["checkpoint_expert_format"] == "fp8"
    assert inventory["converted_expert_banks"] == 0
    for index, (w13, w2) in experts.items():
        mlp = model.layers[index].mlp
        for weight, stored in ((mlp.w13, w13), (mlp.w2, w2)):
            assert isinstance(weight, BlockFP8Weight) and not weight.requires_grad
            assert torch.equal(weight.dequantize(dtype=torch.float32), stored)


@pytest.mark.parametrize("storage", ["fp8", "bf16"])
def test_an_fp8_checkpoint_converts_to_the_other_storages(tmp_path, storage):
    if (
        storage == "fp8"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() < (8, 9)
    ):
        pytest.skip(
            "two-copy FP8 storage feeds E4M3 GEMMs, from compute capability 8.9"
        )
    experts, _ = _write(tmp_path, fp8_experts=True)
    model, inventory = _load(tmp_path, storage=storage)
    assert inventory["expert_storage"] == storage
    # Each layer's w13 and w2 banks change form.
    assert inventory["converted_expert_banks"] == 2 * len(experts)
    for index, (w13, w2) in experts.items():
        mlp = model.layers[index].mlp
        for weight, stored in ((mlp.w13, w13), (mlp.w2, w2)):
            value = (
                weight.dequantize(dtype=torch.float32)
                if storage == "fp8"
                else weight.float()
            )
            # bf16 rounds the decoded values; fp8 requantizes them per row.
            assert _relative(value, stored) < (5e-3 if storage == "bf16" else 6e-2)
    if storage == "fp8":
        assert model.layers[1].mlp.w13.interleaved


def test_lora_factors_start_as_lora_does(tmp_path):
    _write(tmp_path, fp8_experts=True)
    model, _ = _load(tmp_path, lora_rank=8)
    mlp = model.layers[1].mlp
    assert torch.count_nonzero(mlp.lora_gate_up_b) == 0
    assert torch.count_nonzero(mlp.lora_down_b) == 0
    # Kaiming-uniform per expert: fan-in is the factor's input width.
    for factor in (mlp.lora_gate_up_a, mlp.lora_down_a):
        bound = factor.shape[-1] ** -0.5
        assert factor.abs().max() <= bound
        assert factor.abs().max() > 0.5 * bound
    trainable = {
        name for name, value in model.named_parameters() if value.requires_grad
    }
    assert "layers.1.mlp.lora_gate_up_a" in trainable
    assert "layers.1.mlp.w13" not in trainable
