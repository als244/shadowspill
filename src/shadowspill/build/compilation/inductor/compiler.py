"""Capture executable storage and memory facts from Inductor's actual lowering."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

from torch._guards import TracingContext
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.utils import run_and_get_graph_lowering
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from shadowspill.errors import CompilationError
from shadowspill.pytorch.capture.storage import (
    TaskStorageContract,
    capture_task_storage_contract,
)
from shadowspill.task.manifest import ExecutableTaskManifest, validate_value_contract

from ...memory import compiler_memory
from .aliases import _canonicalize_input_alias_outputs
from .alignment import alignment_examples
from .contract import _graph_lowering_contract, _project_callable_contract
from .manifest import _make_manifest
from .outputs import _select_graph_outputs

_GRAPH_LOWERING_CAPTURE_LOCK = threading.Lock()


def compile_graph(
    graph: GraphModule,
    inputs: Sequence[object],
    semantic_contract: TaskStorageContract,
    input_offsets: tuple[tuple[int, int], ...],
    **options: Any,
) -> tuple[Any, ExecutableTaskManifest | None, dict[str, Any] | None]:
    started = time.perf_counter_ns()
    _canonicalize_input_alias_outputs(graph, semantic_contract, tuple(inputs))
    inner = capture_task_storage_contract(graph, tuple(inputs))
    optimized = _project_callable_contract(graph, inner, semantic_contract)
    validate_value_contract(semantic_contract, optimized)
    _ensure_tracing_shape_environment()
    capture_ns = time.perf_counter_ns() - started

    aligned_inputs = alignment_examples(inputs, input_offsets)
    backend: Any = compile_fx_inner
    with _GRAPH_LOWERING_CAPTURE_LOCK:
        compiled, lowerings = run_and_get_graph_lowering(
            lambda: backend(graph, aligned_inputs, **options)
        )
    if len(lowerings) > 1:
        raise CompilationError(
            f"Inductor exposed multiple GraphLowering results: {len(lowerings)}"
        )
    if not lowerings:
        # A warm AOT result carries its code and matching build facts together.
        # The caller rebuilds foreign cache entries missing those facts; there
        # is no independent sidecar cache to reconcile with the executable.
        return compiled, None, None

    lowering = lowerings[0]
    started = time.perf_counter_ns()
    executable = _graph_lowering_contract(lowering, graph, inner, semantic_contract)
    manifest = _make_manifest(
        semantic_contract,
        optimized,
        executable.storage_contract,
        executable.root_allocations,
        capture_ns=capture_ns + time.perf_counter_ns() - started,
    )
    _, outputs = _select_graph_outputs(lowering, graph, inner, semantic_contract)
    leaves = sorted(view.leaf_index for view in semantic_contract.output_views)
    host_outputs = tuple(
        leaf
        for leaf, output in zip(leaves, outputs, strict=True)
        if output.get_device().type == "cpu"
    )
    memory = {
        **compiler_memory(lowering),
        "off_device_output_leaves": host_outputs,
    }
    return compiled, manifest, memory


def _ensure_tracing_shape_environment() -> None:
    context = TracingContext.try_get()
    if (
        context is not None
        and context.fake_mode is not None
        and context.fake_mode.shape_env is None
    ):
        context.fake_mode.shape_env = ShapeEnv()
