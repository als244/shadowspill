"""Both GLM readers use the same bounded initialization protocol as other families."""

from importlib import import_module
from itertools import pairwise

import pytest
import torch

from tests.workloads.test_glm53_checkpoint import _materialized, _write
from tests.workloads.test_glm53_initialization import (
    mixed_checkpoint as mixed_checkpoint,
)
from workloads.models.shared.glm.checkpoint import physical_state


@pytest.mark.parametrize("family", ["glm53", "glm53_flash"])
@pytest.mark.parametrize("fp8", [False, True])
def test_chunk_loading_matches_whole_import_without_assembling_banks(
    tmp_path, monkeypatch, family, fp8
):
    module = import_module(f"workloads.models.{family}.checkpoint")
    config = import_module(f"workloads.models.{family}.config").Config.tiny(
        hidden_size=128
    )
    _write(
        tmp_path,
        fp8_experts=fp8,
        config=config,
        model_type=module.Model,
        name_for=module.source_name,
        prefix=module.GLMCheckpoint.source_prefix.removesuffix("."),
        nested=family == "glm53_flash",
    )
    with module.GLMCheckpoint(tmp_path) as reader:
        model = _materialized(reader.build_model())
        reader.load_into(model)

        def forbidden(*args, **kwargs):
            raise AssertionError("chunk loading assembled checkpoint tensors")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "cat", forbidden)
            patch.setattr(torch, "stack", forbidden)
            for name, expected in physical_state(model).items():
                count = expected.numel()
                actual = torch.empty(count, dtype=expected.dtype)
                cuts = sorted({0, min(7, count), count // 2, max(count - 5, 0), count})
                for left, right in pairwise(cuts):
                    reader.copy_chunk(name, expected.shape, left, actual[left:right])
                assert torch.equal(
                    actual.view(torch.uint8),
                    expected.detach().view(-1).view(torch.uint8),
                ), name


def test_copy_chunk_reads_only_the_requested_expert(tmp_path, monkeypatch):
    from workloads.models.glm53_flash.checkpoint import GLMCheckpoint

    _write(tmp_path, fp8_experts=True)
    with GLMCheckpoint(tmp_path) as reader:
        model = reader.build_model()
        name = "layers.1.mlp.w13._data"
        weight = physical_state(model)[name]
        calls = []
        original = reader.weights.source.tensor

        def record(key, *, meta=False):
            if not meta:
                calls.append(key)
            return original(key, meta=meta)

        monkeypatch.setattr(reader.weights.source, "tensor", record)
        destination = torch.empty(13, dtype=weight.dtype)
        reader.copy_chunk(name, weight.shape, 2 * weight[0].numel() + 3, destination)
        assert calls == ["model.language_model.layers.1.mlp.experts.2.gate_proj.weight"]

        with pytest.raises(ValueError, match="bounds"):
            reader.copy_chunk(name, weight.shape, weight.numel() - 1, destination)
        with pytest.raises(ValueError, match="shape or dtype"):
            reader.copy_chunk(name, (weight.numel(),), 0, destination)
        with pytest.raises(ValueError, match="flat contiguous"):
            reader.copy_chunk(name, weight.shape, 0, destination[::2])


def test_mixed_dense_fp8_and_bf16_checkpoint_chunks(mixed_checkpoint):
    from workloads.models.glm53_flash.checkpoint import GLMCheckpoint

    with GLMCheckpoint(mixed_checkpoint) as reader:
        model = _materialized(
            reader.build_model(lora_rank=8, router_weight_dtype=torch.float32)
        )
        reader.load_into(model)
        for name, expected in physical_state(model).items():
            if name in reader.lora_initializers:
                continue
            actual = torch.empty(expected.numel(), dtype=expected.dtype)
            middle = actual.numel() // 2
            for left, right in ((0, middle), (middle, actual.numel())):
                reader.copy_chunk(name, expected.shape, left, actual[left:right])
            assert torch.equal(
                actual.view(torch.uint8), expected.detach().view(-1).view(torch.uint8)
            ), name
        name = next(iter(reader.lora_initializers))
        with pytest.raises(ValueError, match="adapter initialization"):
            reader.copy_chunk(name, (), 0, torch.empty(1))
