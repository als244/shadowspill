"""Public forward builder: model call to a persistent unprofiled Program."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch._inductor.codecache import torch_key

from shadowspill.artifacts.program import Program
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.partition import PartitionSpec

from .capture import capture_forward_model
from .compilation import compile_task
from .lowering.forward import lower_forward


def build_forward(
    model: nn.Module,
    *,
    example_args: Sequence[Any] = (),
    example_kwargs: Mapping[str, Any] | None = None,
    partition: PartitionSpec = "auto",
    target: str | torch.device = "cuda",
) -> Program:
    """Build fixed implementations without profiling, pools or live training state.

    CPU and meta models are accepted. This initial CUDA backend finalizes
    kernels on one representative GPU. Planning the returned artifact is CPU-only.
    """
    device = resolve_device(target)
    assert device.index is not None
    captured = capture_forward_model(
        model,
        example_args,
        dict(example_kwargs or {}),
        partition=partition,
        device_index=device.index,
    )
    with torch.cuda.device(device):
        unique = {}
        compiled = []
        for artifact in captured.artifacts:
            key = artifact.compatibility_digest
            if key not in unique:
                unique[key] = compile_task(artifact)
            compiled.append(unique[key])
    return lower_forward(
        captured,
        tuple(compiled),
        setup={
            "torch": torch.__version__,
            "torch_build": torch_key().hex(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
    )
