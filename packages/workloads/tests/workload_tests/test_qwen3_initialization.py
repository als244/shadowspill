"""Qwen input ownership and bounded initialization."""

import json
import math

import numpy as np
import torch

from qualification.reference.training.data.calls import Inputs
from shadowspill.execution.tensors import TensorChunk
from shadowspill_workloads.initialization.decoder import Initialize
from shadowspill_workloads.models.qwen3moe.config import PRESETS
from shadowspill_workloads.models.shared.decoder import RotaryEmbedding


def test_initializer_reconstructs_identical_shards_and_rotary_tables():
    config = PRESETS["tiny"]
    initialize = Initialize("qwen3moe", config, seed=41)
    initialize.block_elements = 257

    def fill(name, shape, dtype, width):
        tensor = torch.empty(math.prod(shape), dtype=dtype)
        for start in range(0, tensor.numel(), width):
            initialize(
                TensorChunk(name, shape, dtype, start, tensor.numel()),
                tensor[start : start + width],
            )
        return tensor.reshape(shape)

    rotary = RotaryEmbedding(config.rotary_width, base=config.rope_base, capacity=16)
    for name, shape, dtype in (
        ("blocks.0.moe.w13", (8, 64, 128), torch.bfloat16),
        ("blocks.0.moe.router.weight", (8, 128), torch.bfloat16),
        ("blocks.0.mixer.q_norm.weight", (32,), torch.bfloat16),
        ("rotary.cosine", (16, 32), torch.float32),
        ("rotary.sine", (16, 32), torch.float32),
    ):
        whole = fill(name, shape, dtype, math.prod(shape))
        # Each call is a disjoint part of one global tensor, as for DP shards.
        pieces = fill(name, shape, dtype, 113)
        torch.testing.assert_close(whole, pieces, rtol=0, atol=0)
        if name.startswith("rotary."):
            torch.testing.assert_close(whole, getattr(rotary, name.split(".")[1]))
        elif name.endswith("norm.weight"):
            assert torch.equal(whole, torch.ones_like(whole))
        else:
            assert abs(whole.float().std().item() - config.initializer_range) < 0.002


def test_rank_local_calls_normalize_by_global_tokens_and_restore_cursor(tmp_path):
    values = np.arange(4097, dtype=np.uint32)
    values.tofile(tmp_path / "train.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 8192}))
    options = {
        "data": str(tmp_path),
        "sequence_length": 16,
        "sequences_per_step": 4,
        "sequences_per_example": 2,
    }
    first = Inputs(options, rank=0, world_size=2)
    second = Inputs(options, rank=1, world_size=2)
    for rank, source in enumerate((first, second)):
        examples = next(source)
        tokens = torch.cat([args[0].flatten() for args in examples["args"]])
        assert torch.equal(tokens, torch.arange(rank * 64, (rank + 1) * 64))
        assert all(kwargs["normalizer"] == 1 / 128 for kwargs in examples["kwargs"])
        restored = Inputs(options, rank=rank, world_size=2)
        restored.load_state_dict(source.state_dict())
        torch.testing.assert_close(next(source), next(restored))
