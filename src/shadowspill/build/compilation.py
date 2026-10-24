"""Finalize and package task code and compiler facts through PyTorch AOT."""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch._guards import detect_fake_mode
from torch._inductor import config
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch._inductor.standalone_compile import AOTCompiledArtifact, standalone_compile
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from shadowspill.errors import CompilationError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation.inductor.compiler import (
    _manifest_inner_compile,
    task_compiler_patches,
)
from shadowspill.pytorch.compilation.inductor.manifest import toolchain
from shadowspill.task.manifest import ExecutableTaskManifest
from shadowspill.task.storage import TaskStorageContract

from .memory import compiler_memory


@dataclass(frozen=True)
class BuiltTask:
    manifest: ExecutableTaskManifest
    payload: bytes
    memory: dict[str, Any]


@dataclass
class _TaskCompiler:
    contract: TaskStorageContract

    def __call__(self, graph: Any, inputs: Any, **options: Any) -> Any:
        manifests: list[ExecutableTaskManifest] = []
        memories: list[dict[str, Any]] = []
        inner = _manifest_inner_compile(
            self.contract,
            manifests,
            canonicalize_input_aliases=True,
            on_lowering=lambda lowering: memories.append(compiler_memory(lowering)),
        )
        compiled = inner(graph, inputs, **options)
        if len(manifests) == len(memories) == 1:
            # AOT's cache retains these plain fields with the matching code.
            # A foreign cache entry without them must be rebuilt, not guessed.
            compiled._shadowspill_build = {  # type: ignore[attr-defined]
                "version": 1,
                "manifest": manifests[0].to_dict(),
                "memory": memories[0],
            }
        return compiled


def _package(compiled: Any, contract: TaskStorageContract) -> BuiltTask | None:
    if not isinstance(compiled, AOTCompiledArtifact):
        return None
    serialize = getattr(compiled.inner_fn.compiled_fn, "serialize", None)
    if not callable(serialize):
        return None
    entry = serialize()
    metadata = getattr(entry.compiled_fw.result, "_shadowspill_build", None)
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        return None
    manifest = ExecutableTaskManifest.from_dict(
        metadata["manifest"], semantic_contract=contract, toolchain=toolchain()
    )
    return BuiltTask(manifest, compiled.serialize(), metadata["memory"])


def compile_task(artifact: GraphArtifact) -> BuiltTask:
    # Each forward/backward task is explicit; packaging must not save hidden
    # autograd state merely because its original arguments require gradients.
    inputs: tuple[Any, ...] = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in artifact.example_arguments
    )
    mode = detect_fake_mode(inputs)
    if mode is not None and mode.shape_env is None:
        mode.shape_env = ShapeEnv()

    def invoke() -> BuiltTask | None:
        compiled = standalone_compile(
            artifact.graph_module,
            inputs,
            dynamic_shapes="from_example_inputs",
            options={"inner_compile": _TaskCompiler(artifact.storage_contract)},
            aot=True,
            fake_mode=mode,
        )
        return _package(compiled, artifact.storage_contract)

    with torch.no_grad(), config.patch(task_compiler_patches()):
        result = invoke()
        if result is None:
            # Preserve AOT bundling: force_disable_caches also disables the
            # metadata needed to serialize its code. A fresh build cache avoids
            # foreign entries, and the complete payload outlives this directory.
            with (
                TemporaryDirectory(prefix="shadowspill-build-") as directory,
                temporary_cache_dir(directory),
            ):
                result = invoke()
    if result is None:
        raise CompilationError("task build did not produce portable code and geometry")
    return result
