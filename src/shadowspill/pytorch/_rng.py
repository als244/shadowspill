"""Default random streams shared by preparation, diagnostics and checkpoints."""

from __future__ import annotations

import random
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, cast

import torch


def rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        state["numpy"] = numpy.random.get_state()
    if device.type == "cuda" and cast(Callable[[], bool], torch.cuda.is_initialized)():
        state["accelerator"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy

        numpy.random.set_state(state["numpy"])
    if "accelerator" in state:
        torch.cuda.set_rng_state(state["accelerator"], device)


@contextmanager
def preserve_rng(device: torch.device) -> Iterator[None]:
    """Restore default random streams after setup, including exceptional exits.

    Only the selected CUDA device is included. Explicit torch.Generator objects
    and custom operator RNG state remain owned by the caller.
    """
    state = rng_state(device)
    try:
        yield
    finally:
        restore_rng(state, device)
