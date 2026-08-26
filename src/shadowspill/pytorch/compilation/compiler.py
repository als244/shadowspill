"""Stateless construction of one explicit PyTorch task executable."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensor

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import TaskStorageContract

from .inductor import (
    ExecutableRootAllocation,
    ExecutableTaskManifest,
    compile_explicit_inductor_task,
)


@dataclass(frozen=True, slots=True)
class CompiledTask:
    """One compiled graph and its optional representative arguments."""

    artifact: GraphArtifact
    function: Callable[..., object]
    example_arguments: tuple[object, ...]
    manifest: ExecutableTaskManifest
    execution_provider: str = "torch-inductor"
    graph_node_count: int = 0
    compilation_phase_timings_ns: tuple[tuple[str, int], ...] = ()

    def __call__(self) -> object:
        # AOT artifacts are already explicit forward/backward programs. A
        # second dispatcher-autograd wrapper would create hidden saved state.
        with torch.no_grad():
            return self.function(*self.example_arguments)


@dataclass(frozen=True, slots=True)
class CompiledTaskSet:
    """Selected process-local entrypoints and their storage manifests."""

    functions: dict[str, Callable[..., object]]
    manifests: dict[str, ExecutableTaskManifest]

    @property
    def storage_contracts(self) -> dict[str, TaskStorageContract]:
        return {
            digest: manifest.storage_contract
            for digest, manifest in self.manifests.items()
        }

    @property
    def root_allocations(
        self,
    ) -> dict[str, tuple[ExecutableRootAllocation, ...]]:
        return {
            digest: manifest.root_allocations
            for digest, manifest in self.manifests.items()
        }


def materialize_example_arguments(
    arguments: Sequence[object], *, device_ordinal: int
) -> tuple[object, ...]:
    """Create concrete values while preserving every storage alias and view."""

    cuda_device = torch.device("cuda", device_ordinal)
    storages: dict[tuple[int, str], torch.Tensor] = {}
    results: list[object] = []
    with torch.no_grad():
        for argument in arguments:
            if not isinstance(argument, torch.Tensor):
                results.append(copy.deepcopy(argument))
                continue
            if argument.layout is not torch.strided:
                raise CaptureError(
                    "compiled task examples currently require strided tensors"
                )
            source_storage = argument.untyped_storage()
            target_device = (
                cuda_device if argument.device.type == "cuda" else argument.device
            )
            key = (source_storage._cdata, str(target_device))
            storage = storages.get(key)
            if storage is None:
                storage = torch.empty(
                    source_storage.nbytes(), dtype=torch.uint8, device=target_device
                )
                storage.zero_()
                storages[key] = storage
            tensor = torch.empty(0, dtype=argument.dtype, device=target_device)
            tensor.set_(
                storage.untyped_storage(),
                argument.storage_offset(),
                tuple(argument.shape),
                tuple(argument.stride()),
            )
            tensor.requires_grad_(argument.requires_grad)
            results.append(tensor)
    return tuple(results)


def compile_artifact(
    artifact: GraphArtifact,
    *,
    device_ordinal: int,
    representative_arguments: Sequence[object] | None = None,
) -> CompiledTask:
    """Compile one explicit FX task from its geometry-only task contract."""

    del device_ordinal
    examples = tuple(
        artifact.example_arguments
        if representative_arguments is None
        else representative_arguments
    )
    examples = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in examples
    )
    compilation = compile_explicit_inductor_task(
        artifact.graph_module,
        examples,
        semantic_contract=artifact.storage_contract,
    )
    callable_arguments = (
        () if any(isinstance(value, FakeTensor) for value in examples) else examples
    )
    return CompiledTask(
        artifact=artifact,
        function=compilation.function,
        example_arguments=callable_arguments,
        manifest=compilation.manifest,
        graph_node_count=len(tuple(artifact.graph_module.graph.nodes)),
        compilation_phase_timings_ns=compilation.phase_timings_ns,
    )


__all__ = [
    "CompiledTask",
    "CompiledTaskSet",
    "compile_artifact",
    "materialize_example_arguments",
]
