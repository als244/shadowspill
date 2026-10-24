"""Relevant build and target identity shared by build, profiling and execution."""

from typing import Any

import torch
from torch._inductor.codecache import torch_key


def build_setup(device: torch.device) -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "torch_build": torch_key().hex(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
    }
