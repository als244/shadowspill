"""Relevant build and target identity shared by build, profiling and execution."""

from typing import Any

import torch
from torch._inductor.codecache import torch_key


def build_setup(device: torch.device) -> dict[str, Any]:
    with torch.cuda.device(device):
        return {
            "torch": torch.__version__,
            "torch_build": torch_key().hex(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "blas_backend": str(torch.backends.cuda.preferred_blas_library()),
            "blas_workspace_bytes": torch.backends.cuda.cublas_workspace_size(),
            "blaslt_workspace_bytes": torch.backends.cuda.cublaslt_workspace_size(),
        }
