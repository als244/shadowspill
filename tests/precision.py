"""Explicit low-precision test storage, without initializing CUDA."""

import os

import torch


def low_precision_dtype() -> torch.dtype:
    name = os.environ.get("SHADOWSPILL_TEST_DTYPE", "bfloat16")
    if name not in {"float16", "bfloat16"}:
        raise ValueError("SHADOWSPILL_TEST_DTYPE must be float16 or bfloat16")
    return getattr(torch, name)
