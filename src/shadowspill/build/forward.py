"""Public forward builder: model call to a persistent unprofiled Program."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from shadowspill.artifacts.program import Program
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.partition import PartitionSpec

from .capture import capture_forward_model
from .common import build_setup, compile_artifacts
from .lowering.forward import lower_forward


def build_forward(
    model: nn.Module,
    *,
    example_args: Sequence[Any] = (),
    example_kwargs: Mapping[str, Any] | None = None,
    forward_fn: Callable[..., Any] | None = None,
    partition: PartitionSpec = "auto",
    target: str | torch.device = "cuda",
) -> Program:
    """Build fixed implementations without profiling, pools or live training state.

    CPU and meta models are accepted. This initial CUDA backend finalizes
    kernels on one representative GPU. Planning the returned artifact is CPU-only.
    By default capture calls ``model(*args, **kwargs)``. ``forward_fn`` instead
    receives ``(model, *args, **kwargs)``, preserving registered parameter names.
    """
    device = resolve_device(target)
    assert device.index is not None
    captured = capture_forward_model(
        model,
        example_args,
        dict(example_kwargs or {}),
        forward_fn=forward_fn,
        partition=partition,
        device_index=device.index,
    )
    compiled = compile_artifacts(captured.artifacts, device)
    return lower_forward(
        captured,
        tuple(compiled[item.compatibility_digest] for item in captured.artifacts),
        setup=build_setup(device),
    )
