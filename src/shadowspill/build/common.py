"""Shared implementation finalization for the three computation boundaries."""

from collections.abc import Iterable
from typing import Any

import torch
from torch._inductor.codecache import torch_key

from shadowspill.pytorch.capture.artifacts import GraphArtifact

from .compilation import BuiltTask, compile_task


def compile_artifacts(
    artifacts: Iterable[GraphArtifact], device: torch.device
) -> dict[str, BuiltTask]:
    result: dict[str, BuiltTask] = {}
    with torch.cuda.device(device):
        for artifact in artifacts:
            key = artifact.compatibility_digest
            if key not in result:
                result[key] = compile_task(artifact)
    return result


def build_setup(device: torch.device) -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "torch_build": torch_key().hex(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
    }
