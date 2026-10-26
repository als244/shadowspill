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
from shadowspill.pytorch.dependencies import operator_dependencies
from shadowspill.task.manifest import ExecutableTaskManifest
from shadowspill.task.storage import TaskStorageContract

from .kernel_names import portable_kernel_names
from .memory import compiler_memory

_BUILD_VERSION = 6


@dataclass(frozen=True)
class BuiltTask:
    manifest: ExecutableTaskManifest
    payload: bytes
    memory: dict[str, Any]
    dependencies: tuple[dict[str, str], ...] = ()
    input_offsets: tuple[tuple[int, int], ...] = ()


@dataclass
class _TaskCompiler:
    contract: TaskStorageContract
    input_offsets: tuple[tuple[int, int], ...] = ()
    # Part of AOT's compiler identity, so changed build facts invalidate its cache.
    version: int = _BUILD_VERSION

    def __call__(self, graph: Any, inputs: Any, **options: Any) -> Any:
        manifests: list[ExecutableTaskManifest] = []
        memories: list[dict[str, Any]] = []
        inner = _manifest_inner_compile(
            self.contract,
            manifests,
            canonicalize_input_aliases=True,
            on_lowering=lambda lowering, host_outputs: memories.append(
                {
                    **compiler_memory(lowering),
                    "off_device_output_leaves": host_outputs,
                }
            ),
            input_offsets=self.input_offsets,
        )
        # standalone_compile forces synthetic per-kernel autotuning on. Those
        # buffers cannot represent dependent integer controls. The final build
        # pass instead tunes complete tasks and bundles their selected binaries.
        with (
            config.patch({"triton.autotune_at_compile_time": False}),
            portable_kernel_names(),
        ):
            compiled = inner(graph, inputs, **options)
        if len(manifests) == len(memories) == 1:
            # AOT's cache retains these plain fields with the matching code.
            # A foreign cache entry without them must be rebuilt, not guessed.
            compiled._shadowspill_build = {  # type: ignore[attr-defined]
                "version": self.version,
                "manifest": manifests[0].to_dict(),
                "memory": memories[0],
                "input_offsets": self.input_offsets,
            }
        return compiled


def _package(
    compiled: Any,
    contract: TaskStorageContract,
    input_offsets: tuple[tuple[int, int], ...],
) -> BuiltTask | None:
    if not isinstance(compiled, AOTCompiledArtifact):
        return None
    serialize = getattr(compiled.inner_fn.compiled_fn, "serialize", None)
    if not callable(serialize):
        return None
    entry = serialize()
    metadata = getattr(entry.compiled_fw.result, "_shadowspill_build", None)
    if not isinstance(metadata, dict) or metadata.get("version") != _BUILD_VERSION:
        return None
    if metadata.get("input_offsets", ()) != input_offsets:
        return None  # A cached aligned task cannot serve rank-dependent views.
    manifest = ExecutableTaskManifest.from_dict(
        metadata["manifest"], semantic_contract=contract, toolchain=toolchain()
    )
    return BuiltTask(
        manifest, compiled.serialize(), metadata["memory"], input_offsets=input_offsets
    )


def compile_task(
    artifact: GraphArtifact, *, input_offsets: tuple[tuple[int, int], ...] = ()
) -> BuiltTask:
    dependencies = operator_dependencies(artifact.graph_module)
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
            options={
                "inner_compile": _TaskCompiler(artifact.storage_contract, input_offsets)
            },
            aot=True,
            fake_mode=mode,
        )
        return _package(compiled, artifact.storage_contract, input_offsets)

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
    return BuiltTask(
        result.manifest, result.payload, result.memory, dependencies, input_offsets
    )
