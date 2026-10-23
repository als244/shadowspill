"""Checkpoint-shaped random initialization preserves storage without payload reads."""

import pytest
import torch

pytest.importorskip("mlops")

from mlops.kernels.block_fp8 import quantize_blocks

from tests.workloads.test_glm53_checkpoint import (
    _materialized,
    _write,
    safetensors_torch,
)
from workloads.models.glm53_flash.checkpoint import GLMCheckpoint
from workloads.models.glm53_flash.quickstart import experiment
from workloads.models.shared.glm.checkpoint import physical_state
from workloads.models.shared.glm.initialization import (
    InitializationProgress,
    initialize_random,
)


@pytest.fixture
def mixed_checkpoint(tmp_path):
    _write(tmp_path, fp8_experts=True)
    path = tmp_path / "model.safetensors"
    tensors = safetensors_torch.load_file(str(path))
    for name in tensors:
        if name.endswith(("hc_attn_fn", "hc_ffn_fn", "_conv1d.weight")):
            tensors[name] = tensors[name].bfloat16()
    name = "model.language_model.layers.0.mlp.gate_proj.weight"
    data, scale = quantize_blocks(tensors[name].float().unsqueeze(0))
    tensors[name] = data[0]
    tensors[name.removesuffix(".weight") + ".weight_scale_inv"] = scale[0]
    safetensors_torch.save_file(tensors, str(path))
    return tmp_path


def test_random_default_uses_only_metadata_and_keeps_physical_storage(
    mixed_checkpoint, monkeypatch, capsys
):
    with GLMCheckpoint(mixed_checkpoint) as source:
        declared = source.build_model(lora_rank=8, lora_factor_dtype=torch.bfloat16)
        expected = {
            name: (tuple(value.shape), value.dtype, value.requires_grad)
            for name, value in physical_state(declared).items()
        }
        reader_type = type(source.weights.source)
        original = reader_type.tensor

    def metadata_only(self, name, *, meta=False):
        assert meta, f"unexpected payload read: {name}"
        return original(self, name, meta=meta)

    monkeypatch.setattr(reader_type, "tensor", metadata_only)
    spec = experiment(
        device="cpu",
        checkpoint=mixed_checkpoint,
        lora_rank=8,
        sequence_length=8,
        sequences_per_step=1,
        sequences_per_microbatch=(1,),
    )
    assert spec["metadata"]["initialization"] == "random"
    with spec["context"]():
        model = _materialized(spec["model_factory"]())
        spec["initialize"](model)
    actual = physical_state(model)
    assert {
        name: (tuple(value.shape), value.dtype, value.requires_grad)
        for name, value in actual.items()
    } == expected
    assert any(value.dtype == torch.float8_e4m3fn for value in actual.values())
    for name, value in actual.items():
        assert torch.isfinite(value.float()).all(), name
        if name.endswith(("._scale", ".components.1")):
            assert (value > 0).all(), name
        if "lora" in name and name.endswith("_b"):
            assert torch.count_nonzero(value) == 0, name
    assert model.layers[0].attn_hc.fn.dtype == torch.bfloat16
    assert model.layers[0].self_attn.conv1d.weight.dtype == torch.bfloat16
    output = capsys.readouterr().out
    assert "initialization[random]: starting" in output
    assert f"{len(actual):,}/{len(actual):,} tensors" in output
    assert "ETA 0s" in output


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_seed_is_stable_with_adopted_state_and_writer(mixed_checkpoint, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from shadowspill.pytorch import initialize_storage_

    with GLMCheckpoint(mixed_checkpoint) as source:
        model = _materialized(source.build_model(lora_rank=8))
    initialize_random(model, seed=123, device=device)
    state = physical_state(model)
    expected = {name: value.detach().clone() for name, value in state.items()}
    adopted = next(iter(state))
    initialize_random(
        model,
        seed=123,
        device=device,
        adopted={adopted},
        write_tensor=initialize_storage_,
    )
    for name, value in state.items():
        assert torch.equal(value.float(), expected[name].float()), name
    initialize_random(model, seed=124, device=device, adopted={adopted})
    assert torch.equal(state[adopted].float(), expected[adopted].float())
    assert not torch.equal(
        model.layers[1].mlp.w13._data.float(),
        expected["layers.1.mlp.w13._data"].float(),
    )


def test_checkpoint_mode_loads_actual_weights_and_reports_every_tensor(
    mixed_checkpoint, capsys
):
    spec = experiment(
        device="cpu",
        checkpoint=mixed_checkpoint,
        initialization="checkpoint",
        lora_rank=8,
        sequence_length=8,
        sequences_per_step=1,
        sequences_per_microbatch=(1,),
    )
    with spec["context"]():
        model = _materialized(spec["model_factory"]())
        spec["initialize"](model)
    with GLMCheckpoint(mixed_checkpoint) as source:
        expected = _materialized(source.build_model(lora_rank=8))
        progress = InitializationProgress(expected, mode="checkpoint")
        source.load_into(expected, on_tensor=progress)
        state = physical_state(expected)
        assert progress.done == len(state)
        assert progress.bytes == progress.total
    for name, value in physical_state(model).items():
        if "lora" not in name:
            assert torch.equal(value.float(), state[name].float()), name
    assert "initialization[checkpoint]" in capsys.readouterr().out


@pytest.mark.parametrize(
    "options",
    [
        {"initialization": "typo", "tiny": True},
        {"initialization": "checkpoint", "tiny": True},
    ],
)
def test_invalid_initialization_fails_early(options):
    with pytest.raises(ValueError):
        experiment(device="cpu", **options)
