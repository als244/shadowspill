"""The trainer's pure helpers: which linears Float8 converts and how NF4
blocks a weight too small for torchao's defaults."""

import torch

from baselines.fsdp2.train import float8_filter, nf4_block_sizes


def test_float8_converts_block_linears_the_kernels_take():
    assert float8_filter(torch.nn.Linear(128, 256), "blocks.0.attn.wq")
    assert not float8_filter(torch.nn.Linear(128, 100), "blocks.0.mlp.w1")
    assert not float8_filter(torch.nn.Linear(128, 512), "lm_head")
    assert not float8_filter(torch.nn.Linear(128, 512), "embed")


def test_nf4_blocks_shrink_the_scaler_block_for_small_weights():
    assert nf4_block_sizes(2048 * 2048) == (64, 256)
    assert nf4_block_sizes(64 * 128) == (64, 128)
    assert nf4_block_sizes(64 * 3) == (64, 1)
    assert nf4_block_sizes(100) is None
