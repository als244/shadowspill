"""Compile one graph while capturing the lowering it produces."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torch._guards import TracingContext
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import run_and_get_graph_lowering
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from shadowspill.errors import CompilationError
from shadowspill.pytorch.capture.storage import (
    TaskStorageContract,
    capture_task_storage_contract,
)
from shadowspill.task.manifest import (
    ExecutableTaskManifest,
    validate_value_contract,
)

from .aliases import _canonicalize_input_alias_outputs
from .cache import _fx_graph_cache_key, _load_cached_manifest, _store_cached_manifest
from .contract import _graph_lowering_contract, _project_callable_contract
from .manifest import _make_manifest

# Geometry is an input to planning: it sizes objects and alias extents before a
# task is ever compiled, so the compiler is not free to choose it. Shape padding
# would -- it rewrites a matrix product into pad/product/slice, and a (10, 6)
# gradient then comes back with stride (8, 1), needing 312 bytes and not 240.
#
# The compiler's own promise to keep an output's stride does not hold this back.
# A task compiles through the forward entry point with inference set, and that
# runs the joint-graph passes -- where the padding is introduced -- before the
# outputs are marked visible and their strides recorded. The stride recorded as
# the original one is therefore already padded, and the promise then preserves
# the padding it exists to prevent.
_PINNED_OUTPUT_LAYOUT: Mapping[str, Any] = {"shape_padding": False}

_GRAPH_LOWERING_CAPTURE_LOCK = threading.Lock()
_COMPILATION_PHASE_ORDER = (
    "shadowspill_compiler_input_setup",
    "torch_decomposition_normalization",
    "shadowspill_output_contract_normalization",
    "torch_compiler_configuration",
    "shadowspill_input_alias_normalization",
    "shadowspill_optimized_contract",
    "torch_inductor_core",
    "shadowspill_executable_contract",
    "shadowspill_manifest_assembly",
    "shadowspill_manifest_sidecar",
    "torch_compile_fx_forward_orchestration",
    "shadowspill_callable_wrapper",
)


@dataclass(slots=True)
class _ManifestCompiler:
    semantic_contract: TaskStorageContract
    manifests: list[ExecutableTaskManifest]
    canonicalize_input_aliases: bool
    phase_timings: dict[str, int] | None

    def __call__(
        self,
        optimized_graph: GraphModule,
        optimized_inputs: Sequence[object],
        **options: Any,
    ) -> object:
        alias_ns = self._normalize_input_aliases(optimized_graph, optimized_inputs)
        inner_contract, optimized_contract, contract_ns = (
            self._capture_optimized_contract(optimized_graph, optimized_inputs)
        )
        compiled, graph_lowerings = self._compile_graph(
            optimized_graph, optimized_inputs, options
        )
        cache_key = self._capture_cache_key(compiled)
        self._publish_manifest(
            cache_key,
            graph_lowerings,
            optimized_graph,
            inner_contract,
            optimized_contract,
            alias_ns + contract_ns,
        )
        return compiled

    def _record(self, name: str, started_ns: int) -> int:
        duration = time.perf_counter_ns() - started_ns
        if self.phase_timings is not None:
            self.phase_timings[name] = self.phase_timings.get(name, 0) + duration
        return duration

    def _normalize_input_aliases(
        self,
        graph: GraphModule,
        inputs: Sequence[object],
    ) -> int:
        started_ns = time.perf_counter_ns()
        if self.canonicalize_input_aliases:
            _canonicalize_input_alias_outputs(
                graph, self.semantic_contract, tuple(inputs)
            )
        return self._record("shadowspill_input_alias_normalization", started_ns)

    def _capture_optimized_contract(
        self,
        graph: GraphModule,
        inputs: Sequence[object],
    ) -> tuple[TaskStorageContract, TaskStorageContract, int]:
        started_ns = time.perf_counter_ns()
        inner = capture_task_storage_contract(graph, tuple(inputs))
        optimized = _project_callable_contract(graph, inner, self.semantic_contract)
        validate_value_contract(self.semantic_contract, optimized)
        _ensure_tracing_shape_environment()
        duration = self._record("shadowspill_optimized_contract", started_ns)
        return inner, optimized, duration

    def _compile_graph(
        self,
        graph: GraphModule,
        inputs: Sequence[object],
        options: Mapping[str, object],
    ) -> tuple[object, list[GraphLowering]]:
        started_ns = time.perf_counter_ns()
        inner_backend: Any = compile_fx_inner
        with _GRAPH_LOWERING_CAPTURE_LOCK:
            compiled, graph_lowerings = run_and_get_graph_lowering(
                lambda: inner_backend(graph, inputs, **options)
            )
        self._record("torch_inductor_core", started_ns)
        return compiled, graph_lowerings

    def _capture_cache_key(self, compiled: object) -> str | None:
        started_ns = time.perf_counter_ns()
        cache_key = _fx_graph_cache_key(compiled)
        self._record("shadowspill_manifest_sidecar", started_ns)
        return cache_key

    def _publish_manifest(
        self,
        cache_key: str | None,
        graph_lowerings: list[GraphLowering],
        graph: GraphModule,
        inner_contract: TaskStorageContract,
        optimized_contract: TaskStorageContract,
        capture_ns: int,
    ) -> None:
        if len(graph_lowerings) > 1:
            raise CompilationError(
                "Inductor exposed multiple GraphLowering results: "
                f"observed={len(graph_lowerings)}"
            )
        if graph_lowerings:
            self._publish_graph_manifest(
                cache_key,
                graph_lowerings[0],
                graph,
                inner_contract,
                optimized_contract,
                capture_ns,
            )
        elif cache_key is not None:
            self._publish_cached_manifest(cache_key, optimized_contract, capture_ns)

    def _publish_graph_manifest(
        self,
        cache_key: str | None,
        graph_lowering: GraphLowering,
        graph: GraphModule,
        inner_contract: TaskStorageContract,
        optimized_contract: TaskStorageContract,
        capture_ns: int,
    ) -> None:
        started_ns = time.perf_counter_ns()
        executable = _graph_lowering_contract(
            graph_lowering,
            graph,
            inner_contract,
            self.semantic_contract,
        )
        capture_ns += self._record("shadowspill_executable_contract", started_ns)
        started_ns = time.perf_counter_ns()
        manifest = _make_manifest(
            self.semantic_contract,
            optimized_contract,
            executable.storage_contract,
            executable.root_allocations,
            capture_ns=capture_ns,
        )
        self.manifests.append(manifest)
        self._record("shadowspill_manifest_assembly", started_ns)
        if cache_key is not None:
            started_ns = time.perf_counter_ns()
            _store_cached_manifest(cache_key, manifest)
            self._record("shadowspill_manifest_sidecar", started_ns)

    def _publish_cached_manifest(
        self,
        cache_key: str,
        optimized_contract: TaskStorageContract,
        capture_ns: int,
    ) -> None:
        started_ns = time.perf_counter_ns()
        cached = _load_cached_manifest(
            cache_key,
            self.semantic_contract,
            optimized_contract=optimized_contract,
            capture_ns=capture_ns,
        )
        self._record("shadowspill_manifest_sidecar", started_ns)
        if cached is not None:
            self.manifests.append(cached)


def _ordered_compilation_timings(
    values: Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    """Return non-overlapping compiler phases in a stable public order."""

    ordered = [(name, values.get(name, 0)) for name in _COMPILATION_PHASE_ORDER]
    known = set(_COMPILATION_PHASE_ORDER)
    ordered.extend(
        sorted((name, value) for name, value in values.items() if name not in known)
    )
    return tuple((name, value) for name, value in ordered if value)


def _record_compilation_phase(
    timings: dict[str, int],
    name: str,
    started_ns: int,
) -> int:
    duration = time.perf_counter_ns() - started_ns
    timings[name] = timings.get(name, 0) + duration
    return duration


def _ensure_tracing_shape_environment() -> None:
    problem = TracingContext.try_get()
    if (
        problem is not None
        and problem.fake_mode is not None
        and problem.fake_mode.shape_env is None
    ):
        problem.fake_mode.shape_env = ShapeEnv()


def _manifest_inner_compile(
    semantic_contract: TaskStorageContract,
    manifests: list[ExecutableTaskManifest],
    *,
    canonicalize_input_aliases: bool = False,
    phase_timings: dict[str, int] | None = None,
) -> Callable[..., object]:
    """Return the compiler callback that publishes one physical contract."""

    return _ManifestCompiler(
        semantic_contract,
        manifests,
        canonicalize_input_aliases,
        phase_timings,
    )
