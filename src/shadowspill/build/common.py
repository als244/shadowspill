"""Shared implementation finalization for the three computation boundaries."""

from collections.abc import Iterable, Mapping

import torch

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.toolchain import build_setup as build_setup

from .compilation import BuiltTask, compile_task


def compile_artifacts(
    artifacts: Iterable[GraphArtifact],
    device: torch.device,
    *,
    input_offsets: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
) -> dict[str, BuiltTask]:
    result: dict[str, BuiltTask] = {}
    with torch.cuda.device(device):
        for artifact in artifacts:
            key = artifact.compatibility_digest
            if key not in result:
                result[key] = compile_task(
                    artifact, input_offsets=(input_offsets or {}).get(key, ())
                )
    return result
