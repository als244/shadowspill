"""Shared implementation finalization for the three computation boundaries."""

from collections.abc import Iterable, Mapping

import torch

from shadowspill.errors import CompilationError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.toolchain import build_setup as build_setup

from .compilation import BuiltTask, compile_task
from .compilation.chunking import compile_chunk_task
from .compilation.regions import compile_region_task


def compile_artifacts(
    artifacts: Iterable[GraphArtifact],
    device: torch.device,
    *,
    input_offsets: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
) -> dict[str, BuiltTask]:
    result: dict[str, BuiltTask] = {}
    children: dict[str, BuiltTask] = {}

    def compile_child(artifact: GraphArtifact) -> BuiltTask:
        key = artifact.compatibility_digest
        if key in children:
            return children[key]
        if artifact.regions is not None:
            for child in artifact.regions.children:
                compile_child(child)
            built = compile_region_task(artifact, children)
        elif artifact.chunking is not None:
            for _, child in (
                *artifact.chunking.children,
                *artifact.chunking.accumulating_children,
            ):
                compile_child(child)
            built = compile_chunk_task(artifact, children)
        else:
            built = compile_task(
                artifact, input_offsets=(input_offsets or {}).get(key, ())
            )
        children[key] = built
        return built

    with torch.cuda.device(device):
        for artifact in artifacts:
            key = artifact.compatibility_digest
            if key not in result:
                try:
                    result[key] = compile_child(artifact)
                except Exception as error:
                    raise CompilationError(
                        f"task compilation failed ({artifact.kind}, {key}): {error}",
                        structural_contract=key,
                        task_kind=artifact.kind,
                        operators=artifact.operator_targets,
                    ) from error
    return result
