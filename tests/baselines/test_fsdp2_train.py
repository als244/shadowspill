"""The trainer's pure helpers: which linears Float8 converts."""

import torch

from baselines.fsdp2.train import float8_filter


def test_float8_converts_block_linears_the_kernels_take():
    assert float8_filter(torch.nn.Linear(128, 256), "blocks.0.attn.wq")
    assert not float8_filter(torch.nn.Linear(128, 100), "blocks.0.mlp.w1")
    assert not float8_filter(torch.nn.Linear(128, 512), "lm_head")
    assert not float8_filter(torch.nn.Linear(128, 512), "embed")
