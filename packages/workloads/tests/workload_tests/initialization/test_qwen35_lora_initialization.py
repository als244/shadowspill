"""A real family checkpoint through the shared bounded initializer."""

import json
import math

import pytest
import torch
from safetensors.torch import save_file

from shadowspill.execution.tensors import TensorChunk
from shadowspill_workloads.initialization.decoder import FromCheckpoint
from shadowspill_workloads.lora import configure_lora
from workload_tests.initialization.test_checkpoints import (
    built,
    huggingface_state,
    metadata,
)


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
    with FromCheckpoint("qwen35moe", root, max_seq_len=32, seed=45) as initialize:
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
