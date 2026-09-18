"""The explicit path: a traced graph handed straight to Inductor."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch._guards import TracingContext, detect_fake_mode, tracing
from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import (  # type: ignore[attr-defined]
    compile_fx_forward,
    create_compiler_config_extra,
    select_decomp_table,
)
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from shadowspill.errors import CompilationError
from shadowspill.pytorch.capture.storage import (
    TaskStorageContract,
)
from shadowspill.task.manifest import ExecutableTaskManifest

from .cache import _fx_graph_cache_key, _load_cached_manifest
from .compiler import _PINNED_OUTPUT_LAYOUT, _record_compilation_phase


def _prepare_explicit_inputs(
    example_inputs: Sequence[object],
    timings: dict[str, int],
) -> tuple[FakeTensorMode, tuple[object, ...]]:
    started_ns = time.perf_counter_ns()
    fake_mode = detect_fake_mode(example_inputs)
    if fake_mode is None:
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    if fake_mode.shape_env is None:
        fake_mode.shape_env = ShapeEnv()
    fake_inputs = tuple(
        value
        if isinstance(value, FakeTensor) and value.fake_mode is fake_mode
        else fake_mode.from_tensor(value)
        if isinstance(value, torch.Tensor)
        else value
        for value in example_inputs
    )
    _record_compilation_phase(timings, "shadowspill_compiler_input_setup", started_ns)
    return fake_mode, fake_inputs


def _normalize_explicit_graph(
    graph_module: GraphModule,
    fake_inputs: tuple[object, ...],
    fake_mode: FakeTensorMode,
    timings: dict[str, int],
) -> GraphModule:
    started_ns = time.perf_counter_ns()
    try:
        with V.set_fake_mode(fake_mode), tracing(TracingContext(fake_mode)):
            normalized = make_fx(
                graph_module,
                decomposition_table=select_decomp_table(),
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(*fake_inputs)
    except BaseException as error:
        raise CompilationError(
            f"explicit Inductor task normalization failed: {error}"
        ) from error
    _record_compilation_phase(timings, "torch_decomposition_normalization", started_ns)
    return normalized


def _normalize_explicit_output_contract(
    graph: GraphModule,
    timings: dict[str, int],
) -> tuple[list[object], TreeSpec]:
    started_ns = time.perf_counter_ns()
    output_node = next(node for node in graph.graph.nodes if node.op == "output")
    output_leaves, output_spec = tree_flatten(output_node.args[0])
    output_node.args = (tuple(output_leaves),)
    graph.graph.lint()
    graph.recompile()
    _record_compilation_phase(
        timings, "shadowspill_output_contract_normalization", started_ns
    )
    return output_leaves, output_spec


def _explicit_compiler_config(
    graph: GraphModule,
    timings: dict[str, int],
) -> Any:
    started_ns = time.perf_counter_ns()
    config = create_compiler_config_extra(graph)
    _record_compilation_phase(timings, "torch_compiler_configuration", started_ns)
    return config


def _invoke_explicit_compiler(
    graph: GraphModule,
    fake_inputs: tuple[object, ...],
    fake_mode: FakeTensorMode,
    output_count: int,
    compiler_config: Any,
    inner_compile: Callable[..., object],
    timings: dict[str, int],
) -> object:
    nested_before = sum(timings.values())
    started_ns = time.perf_counter_ns()
    try:
        with (
            V.set_fake_mode(fake_mode),
            tracing(TracingContext(fake_mode)),
            inductor_config.patch(_PINNED_OUTPUT_LAYOUT),
        ):
            compiler = cast(Callable[..., object], compile_fx_forward)
            return compiler(
                graph,
                fake_inputs,
                num_orig_model_outputs=output_count,
                num_example_inputs=len(fake_inputs),
                compiler_config_extra=compiler_config,
                inner_compile=inner_compile,
                is_inference=True,
            )
    finally:
        elapsed = time.perf_counter_ns() - started_ns
        nested_elapsed = sum(timings.values()) - nested_before
        timings["torch_compile_fx_forward_orchestration"] = timings.get(
            "torch_compile_fx_forward_orchestration", 0
        ) + max(0, elapsed - nested_elapsed)


def _compile_with_manifest_regeneration(
    invoke: Callable[[], object],
    manifests: list[ExecutableTaskManifest],
    semantic_contract: TaskStorageContract,
    timings: dict[str, int],
) -> object:
    try:
        compiled = invoke()
    except BaseException as error:
        raise CompilationError(
            f"explicit Inductor task compilation failed: {error}"
        ) from error
    if not manifests:
        _restore_explicit_manifest(compiled, manifests, semantic_contract, timings)
    if not manifests:
        try:
            with inductor_config.patch({"force_disable_caches": True}):
                compiled = invoke()
        except BaseException as error:
            raise CompilationError(
                f"explicit Inductor manifest regeneration failed: {error}"
            ) from error
    if len(manifests) != 1:
        raise CompilationError(
            "explicit Inductor compilation did not expose one root graph: "
            f"observed={len(manifests)}"
        )
    return compiled


def _restore_explicit_manifest(
    compiled: object,
    manifests: list[ExecutableTaskManifest],
    semantic_contract: TaskStorageContract,
    timings: dict[str, int],
) -> None:
    started_ns = time.perf_counter_ns()
    cache_key = _fx_graph_cache_key(compiled)
    if cache_key is not None:
        cached = _load_cached_manifest(
            cache_key,
            semantic_contract,
            optimized_contract=None,
            capture_ns=0,
        )
        if cached is not None:
            manifests.append(cached)
    _record_compilation_phase(timings, "shadowspill_manifest_sidecar", started_ns)


def _unbox_compiled_callable(
    compiled: object,
    output_spec: TreeSpec,
    timings: dict[str, int],
) -> Callable[..., object]:
    started_ns = time.perf_counter_ns()
    compiled_callable = cast(
        Callable[[list[object]], Sequence[object]],
        compiled,
    )

    def unboxed(*arguments: object) -> object:
        values = compiled_callable(list(arguments))
        return tree_unflatten(list(values), output_spec)

    _record_compilation_phase(timings, "shadowspill_callable_wrapper", started_ns)
    return unboxed
