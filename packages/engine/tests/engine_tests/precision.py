"""Explicit low-precision test storage, without initializing CUDA, the
capability a test may ask the device for once it runs, and the rounding step
two BF16 implementations may differ by."""

import math
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


def bf16_step(value: torch.Tensor) -> float:
    """One BF16 rounding step at a tensor's largest magnitude.

    BF16 keeps eight significant bits, so the step is ``2**(e - 7)`` for a
    largest value in ``[2**e, 2**(e + 1))``. An element that is a sum of terms
    as large as the tensor's largest carries their rounding however small the
    element itself comes out, so two correct implementations that round in a
    different order differ by a step or two of those terms, not of the
    element. A comparison with a fixed absolute tolerance fails as soon as the
    values grow past it.
    """

    largest = float(value.detach().float().abs().max())
    return 2.0 ** (math.floor(math.log2(largest)) - 7) if largest else 0.0


def low_precision_dtype() -> torch.dtype:
    name = os.environ.get("SHADOWSPILL_TEST_DTYPE", "bfloat16")
    if name not in {"float16", "bfloat16"}:
        raise ValueError("SHADOWSPILL_TEST_DTYPE must be float16 or bfloat16")
    return getattr(torch, name)
