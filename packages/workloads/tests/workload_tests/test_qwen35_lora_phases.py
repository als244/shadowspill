"""A real family checkpoint through bounded loading and the saved-Plan recipe."""

import json
import math

import pytest
import torch
from safetensors.torch import save_file

from shadowspill.execution.tensors import TensorChunk
from shadowspill_workloads.lora import configure_lora
from shadowspill_workloads.recipes.text.qwen35_lora.initialize import Initialize
from workload_tests.test_checkpoints import built, huggingface_state, metadata


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(123)
    model = built("qwen35moe-tiny", max_seq_len=32)
    root = tmp_path / "base"
    root.mkdir()
    save_file(
        huggingface_state("qwen35moe", model, model.config), root / "model.safetensors"
    )
    (root / "config.json").write_text(json.dumps(metadata("qwen35moe", model.config)))
    return root, model


def test_bounded_initializer_preserves_base_and_adapter_distributions(checkpoint):
    root, base = checkpoint
    expected = dict(base.named_parameters()) | dict(base.named_buffers())
    model = configure_lora(built("qwen35moe-tiny", max_seq_len=32), rank=32, alpha=32)
    options = {"checkpoint": root, "sequence_length": 32, "seed": 45}
    with Initialize(options) as initialize:
        initialize.block_elements = 257
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            values = []
            for width in (tensor.numel(), 113):
                value = torch.empty(tensor.numel(), dtype=tensor.dtype)
                for start in range(0, value.numel(), width):
                    request = TensorChunk(
                        name,
                        tuple(tensor.shape),
                        tensor.dtype,
                        start,
                        math.prod(tensor.shape),
                    )
                    initialize(request, value[start : start + width])
                values.append(value.reshape(tensor.shape))
            torch.testing.assert_close(*values, rtol=0, atol=0)
            if name in expected:
                torch.testing.assert_close(values[0], expected[name], rtol=0, atol=0)
            elif name.endswith("_b"):
                assert values[0].count_nonzero() == 0
            else:
                assert values[0].count_nonzero() > 0
                if name.endswith(".lora_a"):
                    assert values[0].abs().max() <= tensor.shape[-1] ** -0.5 + 0.001
    assert initialize.checkpoint.reader._closed


def test_random_public_initializer_needs_only_configuration(checkpoint, tmp_path):
    from shadowspill_workloads.recipes.text.qwen35_lora.adapter import (
        describe,
        initializer,
    )

    root, base = checkpoint
    (root / "model.safetensors").unlink()
    data = tmp_path / "data"
    data.mkdir()
    (data / "meta.json").write_text(json.dumps({"vocab_size": 32}))
    options = dict(
        checkpoint=root, data=data, initialization="random", seed=17, sequence_length=32
    )
    info = describe(options)
    assert info["initialization"] == "random"
    assert info["frozen_state_id"].startswith("random-v1:")
    expected = base.embed.weight
    request = TensorChunk(
        "embed.weight", tuple(expected.shape), expected.dtype, 0, expected.numel()
    )
    with initializer(options, info) as fill:
        whole = torch.empty(257, dtype=expected.dtype, device="cpu")
        fill(request, whole)
        piece = torch.empty(100, dtype=expected.dtype, device="cpu")
        fill(
            TensorChunk(
                request.name, request.shape, request.dtype, 31, request.storage_elements
            ),
            piece,
        )
    assert torch.equal(piece, whole[31:131])
    assert torch.isfinite(whole).all() and whole.count_nonzero() > 0
