"""Bounded HF imports match whole tensors without assembling model-sized copies."""

import json

import pytest
import torch
from mlops.sequential_moe.precision.safetensors import SafeTensors
from safetensors.torch import save_file

from tests.workloads.test_checkpoints import (
    huggingface_state,
    materialized,
    reader,
)
from tests.workloads.test_checkpoints import metadata as huggingface_metadata
from workloads.models.shared.checkpoint import (
    bank,
    concatenated,
    deinterleaved,
    permuted,
    tensor,
)


def disallow_real_assembly(monkeypatch):
    for name in ("cat", "stack"):
        original = getattr(torch, name)

        def assemble(values, *args, _original=original, **kwargs):
            assert all(value.is_meta for value in values), (
                "assembled a full checkpoint tensor"
            )
            return _original(values, *args, **kwargs)

        monkeypatch.setattr(torch, name, assemble)


@pytest.mark.parametrize("width", [1, 17, 113])
def test_copy_crosses_concat_expert_and_head_boundaries(tmp_path, monkeypatch, width):
    payload = {
        "a": torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
        "b": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) + 100,
        "q": torch.arange(96, dtype=torch.float32).reshape(24, 4),
    }
    save_file(payload, tmp_path / "model.safetensors")
    sources = [
        tensor("a"),
        concatenated("a", "b"),
        bank([concatenated("a", "b")] * 3),
        deinterleaved("q", 3),
    ]
    with SafeTensors(tmp_path) as stream:
        expected = [source.read(stream).float() for source in sources]
        disallow_real_assembly(monkeypatch)
        for source, reference in zip(sources, expected, strict=True):
            result = torch.empty(reference.numel())
            for offset in range(0, result.numel(), width):
                source.copy_into(stream, offset, result[offset : offset + width])
            torch.testing.assert_close(
                result.reshape_as(reference), reference, rtol=0, atol=0
            )


@pytest.mark.parametrize("family", ["qwen35", "qwen35moe"])
def test_qwen_checkpoint_parameters_load_in_executor_sized_chunks(
    tmp_path, monkeypatch, family
):
    changes = {"zero_centered_norm": True, "max_seq_len": 32}
    model = materialized(f"{family}-tiny", **changes)
    config = model.config
    state = huggingface_state(family, model, config)
    save_file(
        {name: value.detach().contiguous() for name, value in state.items()},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(huggingface_metadata(family, config))
    )
    disallow_real_assembly(monkeypatch)
    with reader(family)(tmp_path) as checkpoint:
        for name, expected in model.named_parameters():
            actual = torch.empty_like(expected).view(-1)
            for offset in range(0, actual.numel(), 4093):
                checkpoint.copy_chunk(
                    name, expected.shape, offset, actual[offset : offset + 4093]
                )
            torch.testing.assert_close(
                actual.view_as(expected), expected, rtol=0, atol=0
            )


def test_unknown_transform_refuses_an_unbounded_fallback(tmp_path):
    save_file({"a": torch.ones(8, 4)}, tmp_path / "model.safetensors")
    with SafeTensors(tmp_path) as stream:
        source = permuted("a", lambda value: value.t())
        with pytest.raises(ValueError, match="no bounded copy"):
            source.copy_into(stream, 0, torch.empty(8))
        with pytest.raises(ValueError, match="tensor bounds"):
            tensor("a").copy_into(stream, 31, torch.empty(2))
