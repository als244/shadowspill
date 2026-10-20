"""Explicit low-precision test storage, without initializing CUDA, and the
capability a test may ask the device for once it runs."""

import os

import pytest
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


def require_fp8_gemms() -> None:
    """Skip the calling test on a device without E4M3 GEMMs.

    Two-copy FP8 storage is quantized and multiplied as E4M3, which the
    hardware supports from compute capability 8.9; an earlier device can
    still decode FP8 storage through BF16 GEMMs, so only the E4M3 paths
    are skipped. Called from a test body, not at collection, so the CUDA
    context is created by the test that needs it.
    """

    if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("E4M3 GEMMs need compute capability 8.9 or newer")


def low_precision_dtype() -> torch.dtype:
    name = os.environ.get("SHADOWSPILL_TEST_DTYPE", "bfloat16")
    if name not in {"float16", "bfloat16"}:
        raise ValueError("SHADOWSPILL_TEST_DTYPE must be float16 or bfloat16")
    return getattr(torch, name)
