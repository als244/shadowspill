"""Explicit low-precision test storage, without initializing CUDA."""

import os

import torch

from qualification.precision import detect_device_precision


def select_test_dtype(explicit: str | None = None) -> str:
    """CLI overrides the inherited choice; otherwise detect once in a child."""
    name = explicit or os.environ.get("SHADOWSPILL_TEST_DTYPE")
    if name is None:
        hardware = detect_device_precision(allow_cpu=True)
        name = "bfloat16" if hardware["bf16"] else "float16"
    if name not in {"float16", "bfloat16"}:
        raise ValueError("SHADOWSPILL_TEST_DTYPE must be float16 or bfloat16")
    return name


def low_precision_dtype() -> torch.dtype:
    name = os.environ.get("SHADOWSPILL_TEST_DTYPE", "bfloat16")
    if name not in {"float16", "bfloat16"}:
        raise ValueError("SHADOWSPILL_TEST_DTYPE must be float16 or bfloat16")
    return getattr(torch, name)
