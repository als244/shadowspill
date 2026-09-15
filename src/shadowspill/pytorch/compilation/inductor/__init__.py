"""Narrow PyTorch-version boundary for compiling one explicit task graph.

Export/AOT describes logical values. Inductor may simplify those values before
code generation and thereby change the executable output-alias contract. The outer
``compile_fx`` entrypoint may itself use AOTAutograd and append private saved
outputs, so this adapter projects the optimized inner graph through Inductor's
``user_visible_output_idxs`` metadata before publishing a task manifest.

Contract extraction never executes the compiled graph and never consults
allocator telemetry.

The two entry points are here; :mod:`.compiler` runs Inductor and captures the
lowering, :mod:`.outputs` and :mod:`.roots` read the geometry out of it, and
:mod:`.manifest` is the record they produce.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import compile_fx
from torch.fx import GraphModule

from shadowspill.errors import CompilationError
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module

from .cache import _fx_graph_cache_key, _load_cached_manifest, _store_cached_manifest
from .compiler import (
    _manifest_inner_compile,
    _ordered_compilation_timings,
)
from .explicit import (
    _compile_with_manifest_regeneration,
    _explicit_compiler_config,
    _invoke_explicit_compiler,
    _normalize_explicit_graph,
    _normalize_explicit_output_contract,
    _prepare_explicit_inputs,
    _unbox_compiled_callable,
)
from .manifest import (
    ExecutableRootAllocation,
    ExecutableTaskManifest,
    InductorCompilation,
)


def compile_inductor_task(
    graph_module: GraphModule,
    example_inputs: Sequence[object],
    *,
    semantic_contract: TaskStorageContract,
) -> InductorCompilation:
    """Compile and capture the callable-visible optimized output contract."""

    manifests: list[ExecutableTaskManifest] = []
    compilation_started = time.perf_counter_ns()
    source_graph = copy_graph_module(graph_module)
    inner_compile = _manifest_inner_compile(semantic_contract, manifests)

    def invoke_compiler() -> Any:
        compiler: Any = compile_fx
        return compiler(
            copy_graph_module(source_graph),
            list(example_inputs),
            inner_compile=inner_compile,
        )

    try:
        compiled = invoke_compiler()
    except BaseException as exc:
        raise CompilationError(f"Inductor task compilation failed: {exc}") from exc
    if not manifests:
        cache_key = _fx_graph_cache_key(compiled)
        if cache_key is not None:
            cached = _load_cached_manifest(
                cache_key,
                semantic_contract,
                optimized_contract=None,
                capture_ns=time.perf_counter_ns() - compilation_started,
            )
            if cached is not None:
                manifests.append(cached)
        if not manifests:
            # A compiler cache created outside ShadowSpill has no executable
            # storage sidecar. Recompile once without AOT/FX caches so
            # GraphLowering can publish the contract, then seed the sidecar for
            # every later process. Never infer physical aliases from the cached
            # callable or allocator telemetry.
            try:
                with inductor_config.patch({"force_disable_caches": True}):
                    compiled = invoke_compiler()
            except BaseException as exc:
                raise CompilationError(
                    f"Inductor task manifest regeneration failed: {exc}"
                ) from exc
            if len(manifests) == 1 and cache_key is not None:
                _store_cached_manifest(cache_key, manifests[0])
    if len(manifests) != 1:
        raise CompilationError(
            "Inductor task compilation did not expose one optimized root graph: "
            f"observed={len(manifests)}"
        )
    return InductorCompilation(compiled, manifests[0])


def compile_explicit_inductor_task(
    graph_module: GraphModule,
    example_inputs: Sequence[object],
    *,
    semantic_contract: TaskStorageContract,
) -> InductorCompilation:
    """Compile one explicit task without a second AOTAutograd pass."""

    timings: dict[str, int] = {}
    fake_mode, fake_inputs = _prepare_explicit_inputs(example_inputs, timings)
    normalized = _normalize_explicit_graph(
        graph_module, fake_inputs, fake_mode, timings
    )
    output_leaves, output_spec = _normalize_explicit_output_contract(
        normalized, timings
    )
    manifests: list[ExecutableTaskManifest] = []
    inner_compile = _manifest_inner_compile(
        semantic_contract,
        manifests,
        canonicalize_input_aliases=True,
        phase_timings=timings,
    )
    compiler_config = _explicit_compiler_config(normalized, timings)

    def invoke() -> object:
        return _invoke_explicit_compiler(
            normalized,
            fake_inputs,
            fake_mode,
            len(output_leaves),
            compiler_config,
            inner_compile,
            timings,
        )

    compiled = _compile_with_manifest_regeneration(
        invoke, manifests, semantic_contract, timings
    )
    unboxed = _unbox_compiled_callable(compiled, output_spec, timings)
    return InductorCompilation(
        unboxed,
        manifests[0],
        _ordered_compilation_timings(timings),
    )


__all__ = [
    "ExecutableRootAllocation",
    "ExecutableTaskManifest",
    "InductorCompilation",
    "compile_explicit_inductor_task",
    "compile_inductor_task",
]
