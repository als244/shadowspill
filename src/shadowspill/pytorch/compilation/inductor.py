"""Narrow PyTorch-version boundary for compiling one explicit task graph.

Export/AOT describes logical values. Inductor may simplify those values before
code generation and thereby change the executable output-alias ABI. The outer
``compile_fx`` entrypoint may itself use AOTAutograd and append private saved
outputs, so this adapter projects the optimized inner graph through Inductor's
``user_visible_output_idxs`` metadata before publishing a task manifest.

Contract extraction never executes the compiled graph and never consults
allocator telemetry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch._guards import TracingContext, detect_fake_mode, tracing
from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import (  # type: ignore[attr-defined]
    compile_fx,
    compile_fx_forward,
    compile_fx_inner,
    create_compiler_config_extra,
    select_decomp_table,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import run_and_get_graph_lowering
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from shadowspill.pytorch.capture.storage import (
    MutationBinding,
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
    capture_task_storage_contract,
)
from shadowspill.pytorch.compilation.inductor_manifest import (
    CachedTaskManifest,
    load_task_manifest,
    store_task_manifest,
)
from shadowspill.pytorch.contracts import CaptureError, CompilationError


@dataclass(frozen=True, slots=True)
class ExecutableRootAllocation:
    """Compiler-owned allocation extent for one executable storage root."""

    root_id: int
    requested_bytes: int

    def __post_init__(self) -> None:
        if self.root_id < 0 or self.requested_bytes < 0:
            raise ValueError("executable root allocation fields must be non-negative")

    def identity(self) -> dict[str, int]:
        return {
            "root_id": self.root_id,
            "requested_bytes": self.requested_bytes,
        }


@dataclass(frozen=True, slots=True)
class ExecutableTaskManifest:
    """Offline storage ABI emitted for one optimized compiled task."""

    semantic_contract_digest: str
    storage_contract: TaskStorageContract
    contract_capture_ns: int
    compatibility_digest: str
    optimized_storage_contract: TaskStorageContract | None = None
    root_allocations: tuple[ExecutableRootAllocation, ...] = ()

    def __post_init__(self) -> None:
        if len(self.semantic_contract_digest) != 64:
            raise ValueError("semantic contract digest must be SHA-256")
        if self.contract_capture_ns < 0:
            raise ValueError("executable contract timing must be non-negative")
        if len(self.compatibility_digest) != 64:
            raise ValueError("executable manifest digest must be SHA-256")
        if tuple(item.root_id for item in self.root_allocations) != tuple(
            range(len(self.storage_contract.roots))
        ):
            raise ValueError(
                "executable root allocations must have contiguous root indices"
            )
        for root, allocation in zip(
            self.storage_contract.roots, self.root_allocations, strict=True
        ):
            if root.kind is StorageRootKind.INPUT and allocation.requested_bytes:
                raise ValueError("input executable root cannot allocate storage")
            if (
                root.kind is StorageRootKind.FRESH
                and allocation.requested_bytes < root.minimum_span_bytes
            ):
                raise ValueError(
                    "executable root allocation is smaller than its output views"
                )

    def identity(self) -> dict[str, object]:
        return {
            "semantic_contract_digest": self.semantic_contract_digest,
            "optimized_storage_contract": (
                self.optimized_storage_contract.identity()
                if self.optimized_storage_contract is not None
                else self.storage_contract.identity()
            ),
            "storage_contract": self.storage_contract.identity(),
            "storage_contract_digest": self.storage_contract.compatibility_digest,
            "root_allocations": [item.identity() for item in self.root_allocations],
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the compiler-owned storage ABI for a profile sidecar."""

        if self.optimized_storage_contract is None:
            raise ValueError("compiled task manifest has no optimized contract")
        return {
            "semantic_contract_digest": self.semantic_contract_digest,
            "storage_contract": self.storage_contract.to_dict(),
            "contract_capture_ns": self.contract_capture_ns,
            "compatibility_digest": self.compatibility_digest,
            "optimized_storage_contract": (self.optimized_storage_contract.to_dict()),
            "root_allocations": [item.identity() for item in self.root_allocations],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        semantic_contract: TaskStorageContract,
    ) -> ExecutableTaskManifest:
        """Restore and fully validate one cached compiler storage ABI."""

        expected = {
            "semantic_contract_digest",
            "storage_contract",
            "contract_capture_ns",
            "compatibility_digest",
            "optimized_storage_contract",
            "root_allocations",
        }
        if set(payload) != expected:
            raise ValueError("compiled task manifest fields differ from schema")
        if payload["semantic_contract_digest"] != (
            semantic_contract.compatibility_digest
        ):
            raise ValueError("compiled task manifest has the wrong semantic ABI")
        executable = payload["storage_contract"]
        optimized = payload["optimized_storage_contract"]
        capture_ns = payload["contract_capture_ns"]
        declared_digest = payload["compatibility_digest"]
        raw_allocations = payload["root_allocations"]
        if not isinstance(executable, dict) or not isinstance(optimized, dict):
            raise ValueError("compiled task manifest contracts must be objects")
        if (
            not isinstance(capture_ns, int)
            or isinstance(capture_ns, bool)
            or capture_ns < 0
        ):
            raise ValueError("compiled task manifest timing is invalid")
        if not isinstance(declared_digest, str):
            raise ValueError("compiled task manifest digest is invalid")
        if not isinstance(raw_allocations, list):
            raise ValueError("compiled task root allocations must be a list")
        allocations: list[ExecutableRootAllocation] = []
        for item in raw_allocations:
            if not isinstance(item, dict) or set(item) != {
                "root_id",
                "requested_bytes",
            }:
                raise ValueError("compiled task root allocation is invalid")
            root_id = item["root_id"]
            requested_bytes = item["requested_bytes"]
            if (
                not isinstance(root_id, int)
                or isinstance(root_id, bool)
                or not isinstance(requested_bytes, int)
                or isinstance(requested_bytes, bool)
            ):
                raise ValueError("compiled task root allocation is invalid")
            allocations.append(ExecutableRootAllocation(root_id, requested_bytes))
        restored = _make_manifest(
            semantic_contract,
            TaskStorageContract.from_dict(optimized),
            TaskStorageContract.from_dict(executable),
            tuple(allocations),
            capture_ns=capture_ns,
        )
        if restored.compatibility_digest != declared_digest:
            raise ValueError("compiled task manifest digest does not match")
        return restored


@dataclass(frozen=True, slots=True)
class InductorCompilation:
    """Callable and the optimized storage contract it actually implements."""

    function: Callable[..., object]
    manifest: ExecutableTaskManifest
    phase_timings_ns: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _LoweredOutput:
    semantic_view: OutputView
    optimized_view: OutputView
    provenance: StorageRoot
    root_name: str
    offset_bytes: int
    span_bytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class _GraphLoweringManifest:
    storage_contract: TaskStorageContract
    root_allocations: tuple[ExecutableRootAllocation, ...]


_GRAPH_LOWERING_CAPTURE_LOCK = threading.Lock()
_COMPILATION_PHASE_ORDER = (
    "shadowspill_compiler_input_setup",
    "torch_decomposition_normalization",
    "shadowspill_output_abi_normalization",
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
        _validate_value_abi(self.semantic_contract, optimized)
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


def _ensure_tracing_shape_environment() -> None:
    context = TracingContext.try_get()
    if (
        context is not None
        and context.fake_mode is not None
        and context.fake_mode.shape_env is None
    ):
        context.fake_mode.shape_env = ShapeEnv()


def _manifest_inner_compile(
    semantic_contract: TaskStorageContract,
    manifests: list[ExecutableTaskManifest],
    *,
    canonicalize_input_aliases: bool = False,
    phase_timings: dict[str, int] | None = None,
) -> Callable[..., object]:
    """Return the compiler callback that publishes one physical ABI."""

    return _ManifestCompiler(
        semantic_contract,
        manifests,
        canonicalize_input_aliases,
        phase_timings,
    )


def compile_inductor_task(
    graph_module: GraphModule,
    example_inputs: Sequence[object],
    *,
    semantic_contract: TaskStorageContract,
) -> InductorCompilation:
    """Compile and capture the callable-visible optimized output ABI."""

    manifests: list[ExecutableTaskManifest] = []
    compilation_started = time.perf_counter_ns()
    source_graph = copy.deepcopy(graph_module)
    inner_compile = _manifest_inner_compile(semantic_contract, manifests)

    def invoke_compiler() -> Any:
        compiler: Any = compile_fx
        return compiler(
            copy.deepcopy(source_graph),
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
    output_leaves, output_spec = _normalize_explicit_output_abi(normalized, timings)
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


def _record_compilation_phase(
    timings: dict[str, int],
    name: str,
    started_ns: int,
) -> int:
    duration = time.perf_counter_ns() - started_ns
    timings[name] = timings.get(name, 0) + duration
    return duration


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


def _normalize_explicit_output_abi(
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
        timings, "shadowspill_output_abi_normalization", started_ns
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
        with V.set_fake_mode(fake_mode), tracing(TracingContext(fake_mode)):
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


def _canonicalize_input_alias_outputs(
    graph_module: GraphModule,
    contract: TaskStorageContract,
    example_inputs: tuple[object, ...],
) -> None:
    """Make every declared input-alias return explicit in the FX output ABI.

    In-place operators may return their mutated argument, but direct Inductor
    lowering can otherwise materialize that return as a second output buffer.
    Publishing the input (or its exact view) expresses the already-declared
    alias without adding a copy or changing the mutating operation itself.
    """

    output_node = next(node for node in graph_module.graph.nodes if node.op == "output")
    leaves, spec = tree_flatten(output_node.args[0])
    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    root_by_id = {root.root_id: root for root in contract.roots}
    view_by_leaf = {view.leaf_index: view for view in contract.output_views}
    changed = False
    for leaf_index, leaf in enumerate(leaves):
        view = view_by_leaf.get(leaf_index)
        if view is None:
            continue
        root = root_by_id[view.root_id]
        if root.kind is not StorageRootKind.INPUT:
            continue
        replacement = _input_alias_output(
            graph_module,
            output_node,
            placeholders,
            example_inputs,
            root,
            view,
        )
        if replacement is not leaf:
            leaves[leaf_index] = replacement
            changed = True
    if changed:
        output_node.args = (tree_unflatten(leaves, spec),)
        graph_module.graph.lint()
        graph_module.recompile()


def _input_alias_output(
    graph_module: GraphModule,
    output_node: Node,
    placeholders: tuple[Node, ...],
    example_inputs: tuple[object, ...],
    root: StorageRoot,
    view: OutputView,
) -> Node:
    source_position = root.source_input
    if source_position is None:
        raise AssertionError("input storage root omitted its source position")
    source = example_inputs[source_position]
    if not isinstance(source, torch.Tensor):
        raise CaptureError("compiled input alias refers to a non-tensor argument")
    _validate_input_alias_view(view, source, source_position)
    itemsize = source.element_size()
    placeholder = placeholders[source_position]
    source_offset_bytes = int(source.storage_offset()) * itemsize
    if (
        view.shape == tuple(int(value) for value in source.shape)
        and view.stride == tuple(int(value) for value in source.stride())
        and view.offset_bytes == source_offset_bytes
    ):
        return placeholder
    with graph_module.graph.inserting_before(output_node):
        return graph_module.graph.call_function(
            torch.ops.aten.as_strided.default,
            args=(
                placeholder,
                view.shape,
                view.stride,
                view.offset_bytes // itemsize,
            ),
        )


def _validate_input_alias_view(
    view: OutputView,
    source: torch.Tensor,
    source_position: int,
) -> None:
    if view.dtype != str(source.dtype):
        raise CaptureError(
            "direct compilation cannot express a dtype-changing input view: "
            f"leaf={view.leaf_index}, input={source_position}, "
            f"source={source.dtype}, output={view.dtype}"
        )
    itemsize = source.element_size()
    if view.offset_bytes % itemsize:
        raise CaptureError(
            "input-alias output offset is not element aligned: "
            f"leaf={view.leaf_index}, bytes={view.offset_bytes}, itemsize={itemsize}"
        )


def _make_manifest(
    semantic_contract: TaskStorageContract,
    optimized_contract: TaskStorageContract,
    executable_contract: TaskStorageContract,
    root_allocations: tuple[ExecutableRootAllocation, ...],
    *,
    capture_ns: int,
) -> ExecutableTaskManifest:
    _validate_value_abi(semantic_contract, optimized_contract)
    _validate_value_abi(semantic_contract, executable_contract)
    identity = {
        "semantic_contract_digest": semantic_contract.compatibility_digest,
        "optimized_storage_contract": optimized_contract.identity(),
        "optimized_storage_contract_digest": optimized_contract.compatibility_digest,
        "storage_contract": executable_contract.identity(),
        "storage_contract_digest": executable_contract.compatibility_digest,
        "root_allocations": [item.identity() for item in root_allocations],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return ExecutableTaskManifest(
        semantic_contract.compatibility_digest,
        executable_contract,
        capture_ns,
        hashlib.sha256(encoded.encode()).hexdigest(),
        optimized_contract,
        root_allocations,
    )


def _fx_graph_cache_key(compiled: object) -> str | None:
    value = getattr(compiled, "_fx_graph_cache_key", None)
    return value if isinstance(value, str) and value else None


def _load_cached_manifest(
    cache_key: str,
    semantic_contract: TaskStorageContract,
    *,
    optimized_contract: TaskStorageContract | None,
    capture_ns: int,
) -> ExecutableTaskManifest | None:
    cached = load_task_manifest(cache_key, semantic_contract.compatibility_digest)
    if cached is None:
        return None
    if (
        optimized_contract is not None
        and optimized_contract.compatibility_digest
        != cached.optimized_storage_contract.compatibility_digest
    ):
        return None
    try:
        manifest = _make_manifest(
            semantic_contract,
            cached.optimized_storage_contract,
            cached.storage_contract,
            tuple(
                ExecutableRootAllocation(root_id, requested_bytes)
                for root_id, requested_bytes in enumerate(cached.root_allocation_bytes)
            ),
            capture_ns=capture_ns,
        )
    except (CaptureError, CompilationError, ValueError):
        return None
    return (
        manifest
        if manifest.compatibility_digest == cached.compatibility_digest
        else None
    )


def _store_cached_manifest(
    cache_key: str,
    manifest: ExecutableTaskManifest,
) -> None:
    optimized = manifest.optimized_storage_contract
    if optimized is None:
        raise AssertionError("compiled task manifest omitted its optimized contract")
    try:
        store_task_manifest(
            cache_key,
            manifest.semantic_contract_digest,
            CachedTaskManifest(
                optimized,
                manifest.storage_contract,
                tuple(
                    allocation.requested_bytes
                    for allocation in manifest.root_allocations
                ),
                manifest.compatibility_digest,
            ),
        )
    except OSError:
        # The compiler cache is an optimization. The current process already
        # owns a complete manifest and remains correct if its cache directory
        # is read-only or disappears concurrently.
        return


def _project_callable_contract(
    optimized_graph: GraphModule,
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> TaskStorageContract:
    """Remove compiler-private saved outputs using Inductor's ABI metadata."""

    roots, output_views = _project_visible_outputs(
        optimized_graph,
        inner_contract,
        semantic_contract,
    )
    mutations = _project_callable_mutations(
        semantic_contract.mutations,
        roots,
        output_views,
    )
    return _make_storage_contract(roots, output_views, mutations)


def _project_visible_outputs(
    optimized_graph: GraphModule,
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> tuple[tuple[StorageRoot, ...], tuple[OutputView, ...]]:
    visible = _visible_output_indices(optimized_graph)
    inner_view_by_leaf = {view.leaf_index: view for view in inner_contract.output_views}
    visible_views = tuple(
        inner_view_by_leaf[index] for index in visible if index in inner_view_by_leaf
    )
    semantic_views = tuple(
        sorted(semantic_contract.output_views, key=lambda view: view.leaf_index)
    )
    if len(visible_views) != len(semantic_views):
        raise CaptureError(
            "Inductor callable-visible tensor output count changed: "
            f"semantic={len(semantic_views)}, executable={len(visible_views)}, "
            f"visible_indices={visible}"
        )
    inner_root_by_id = {root.root_id: root for root in inner_contract.roots}
    selected_root_ids = tuple(dict.fromkeys(view.root_id for view in visible_views))
    root_index = {original: index for index, original in enumerate(selected_root_ids)}
    roots = tuple(
        _copy_root(inner_root_by_id[original], root_index[original])
        for original in selected_root_ids
    )
    output_views = tuple(
        _copy_view(executable, semantic.leaf_index, root_index[executable.root_id])
        for executable, semantic in zip(visible_views, semantic_views, strict=True)
    )
    return roots, output_views


def _project_callable_mutations(
    mutations: tuple[MutationBinding, ...],
    roots: tuple[StorageRoot, ...],
    output_views: tuple[OutputView, ...],
) -> tuple[MutationBinding, ...]:
    view_by_leaf = {view.leaf_index: view for view in output_views}
    root_by_id = {root.root_id: root for root in roots}
    for mutation in mutations:
        leaf = mutation.replacement_output_leaf
        if leaf is not None:
            view = view_by_leaf.get(leaf)
            if view is None:
                raise CaptureError(
                    "Inductor removed a functional mutation replacement: "
                    f"leaf={leaf}, target={mutation.argument_name}"
                )
            root = root_by_id[view.root_id]
            if (
                root.kind is StorageRootKind.INPUT
                and root.source_input != mutation.input_position
            ):
                raise CaptureError(
                    "functional mutation replacement aliases another input: "
                    f"leaf={leaf}, expected={mutation.input_position}, "
                    f"actual={root.source_input}"
                )
    return mutations


def _graph_lowering_contract(
    graph: GraphLowering,
    optimized_graph: GraphModule,
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> _GraphLoweringManifest:
    """Project Inductor's returned buffers into an executable storage ABI."""

    visible, outputs = _select_graph_outputs(
        graph, optimized_graph, inner_contract, semantic_contract
    )
    input_position_by_name = _graph_input_positions(graph, optimized_graph)
    records = _lower_graph_outputs(
        graph,
        outputs,
        visible,
        inner_contract,
        semantic_contract,
    )
    roots, allocations = _build_executable_roots(graph, records, input_position_by_name)
    output_views = _lowered_output_views(records, roots)
    mutations = _project_mutations(semantic_contract, roots, output_views)
    contract = _make_storage_contract(roots, output_views, mutations)
    return _GraphLoweringManifest(contract, allocations)


def _select_graph_outputs(
    graph: GraphLowering,
    optimized_graph: GraphModule,
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    visible = _visible_output_indices(optimized_graph)
    if graph.graph_outputs is None:
        raise CaptureError("Inductor GraphLowering omitted task outputs")
    if any(index >= len(graph.graph_outputs) for index in visible):
        raise CaptureError(
            "Inductor callable-visible output index exceeds GraphLowering ABI"
        )
    inner_leaves = {view.leaf_index for view in inner_contract.output_views}
    outputs = tuple(
        graph.graph_outputs[index] for index in visible if index in inner_leaves
    )
    if len(outputs) != len(semantic_contract.output_views):
        raise CaptureError(
            "GraphLowering callable-visible tensor output count changed: "
            f"semantic={len(semantic_contract.output_views)}, "
            f"executable={len(outputs)}"
        )
    return visible, outputs


def _graph_input_positions(
    graph: GraphLowering,
    optimized_graph: GraphModule,
) -> dict[str, int]:
    placeholders = tuple(
        node for node in optimized_graph.graph.nodes if node.op == "placeholder"
    )
    return {
        node.name: index
        for index, node in enumerate(placeholders)
        if node.name in graph.graph_inputs
    }


def _lower_graph_outputs(
    graph: GraphLowering,
    outputs: tuple[Any, ...],
    visible: tuple[int, ...],
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> tuple[_LoweredOutput, ...]:
    inner_view_by_leaf = {view.leaf_index: view for view in inner_contract.output_views}
    optimized_views = tuple(
        inner_view_by_leaf[index] for index in visible if index in inner_view_by_leaf
    )
    semantic_views = tuple(
        sorted(semantic_contract.output_views, key=lambda view: view.leaf_index)
    )
    root_by_id = {root.root_id: root for root in inner_contract.roots}
    return tuple(
        _lower_graph_output(
            graph,
            semantic_view,
            optimized_view,
            output,
            root_by_id[optimized_view.root_id],
        )
        for semantic_view, optimized_view, output in zip(
            semantic_views, optimized_views, outputs, strict=True
        )
    )


def _lower_graph_output(
    graph: GraphLowering,
    semantic_view: OutputView,
    optimized_view: OutputView,
    output: Any,
    provenance: StorageRoot,
) -> _LoweredOutput:
    if not output.has_tensor_output():
        raise CaptureError(
            "GraphLowering replaced a tensor output with a non-tensor value: "
            f"leaf={semantic_view.leaf_index}"
        )
    root_name, shape, stride, dtype, offset_bytes, span_bytes = _graph_output_geometry(
        graph, semantic_view, output
    )
    _validate_graph_output_geometry(semantic_view, shape, stride, dtype)
    return _LoweredOutput(
        semantic_view=semantic_view,
        optimized_view=optimized_view,
        provenance=provenance,
        root_name=root_name,
        offset_bytes=offset_bytes,
        span_bytes=span_bytes,
        shape=shape,
        stride=stride,
        dtype=str(dtype),
    )


def _graph_output_geometry(
    graph: GraphLowering,
    semantic_view: OutputView,
    output: Any,
) -> tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype, int, int]:
    try:
        root_name = str(output.get_name())
        layout = output.get_layout()
        dtype = output.get_dtype()
        shape = tuple(
            _static_int(graph, value, "output shape") for value in output.get_size()
        )
        stride = tuple(
            _static_int(graph, value, "output stride") for value in output.get_stride()
        )
        offset_elements = _static_int(graph, layout.offset, "output offset")
    except (AttributeError, NotImplementedError, TypeError) as error:
        raise CaptureError(
            "Inductor GraphLowering output has no concrete strided layout: "
            f"leaf={semantic_view.leaf_index}, type={type(output).__name__}"
        ) from error
    if offset_elements < 0 or any(value < 0 for value in (*shape, *stride)):
        raise CaptureError(
            "Inductor GraphLowering produced a negative output geometry: "
            f"leaf={semantic_view.leaf_index}"
        )
    item_size = torch.empty((), device="meta", dtype=dtype).element_size()
    return (
        root_name,
        shape,
        stride,
        dtype,
        offset_elements * item_size,
        _span_bytes(shape, stride, item_size),
    )


def _validate_graph_output_geometry(
    expected: OutputView,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    significant_strides_match = all(
        extent <= 1 or actual == expected_stride
        for extent, actual, expected_stride in zip(
            shape, stride, expected.stride, strict=True
        )
    )
    expected_geometry = (
        expected.shape,
        expected.stride,
        expected.dtype,
        expected.layout,
    )
    actual_geometry = (shape, stride, str(dtype), str(torch.strided))
    if (
        shape != expected.shape
        or str(dtype) != expected.dtype
        or str(torch.strided) != expected.layout
        or not significant_strides_match
    ):
        raise CaptureError(
            "GraphLowering changed task output geometry: "
            f"leaf={expected.leaf_index}, expected={expected_geometry}, "
            f"actual={actual_geometry}"
        )


def _build_executable_roots(
    graph: GraphLowering,
    records: tuple[_LoweredOutput, ...],
    input_position_by_name: Mapping[str, int],
) -> tuple[tuple[StorageRoot, ...], tuple[ExecutableRootAllocation, ...]]:
    root_names = tuple(dict.fromkeys(record.root_name for record in records))
    root_id_by_name = {name: index for index, name in enumerate(root_names)}
    roots: list[StorageRoot] = []
    allocations: list[ExecutableRootAllocation] = []
    for name in root_names:
        members = tuple(record for record in records if record.root_name == name)
        root, allocation = _build_executable_root(
            graph,
            name,
            root_id_by_name[name],
            members,
            input_position_by_name.get(name),
        )
        roots.append(root)
        allocations.append(allocation)
    return tuple(roots), tuple(allocations)


def _build_executable_root(
    graph: GraphLowering,
    name: str,
    root_id: int,
    members: tuple[_LoweredOutput, ...],
    source_input: int | None,
) -> tuple[StorageRoot, ExecutableRootAllocation]:
    minimum_span = max(record.offset_bytes + record.span_bytes for record in members)
    if source_input is not None:
        return (
            StorageRoot(
                root_id,
                StorageRootKind.INPUT,
                source_input,
                None,
                None,
                None,
                minimum_span,
            ),
            ExecutableRootAllocation(root_id, 0),
        )
    allocation_bytes = _graph_buffer_extent(graph, name)
    if allocation_bytes < minimum_span:
        raise CaptureError(
            "Inductor output allocation is smaller than its returned views: "
            f"root={name}, allocation={allocation_bytes}, "
            f"minimum_span={minimum_span}"
        )
    provenance = members[0].provenance
    return (
        StorageRoot(
            root_id,
            StorageRootKind.FRESH,
            None,
            provenance.producer_node or f"inductor_{name}",
            provenance.producer_target or "inductor.output_buffer",
            provenance.producer_result or 0,
            minimum_span,
        ),
        ExecutableRootAllocation(root_id, allocation_bytes),
    )


def _graph_buffer_extent(graph: GraphLowering, name: str) -> int:
    try:
        buffer: Any = graph.get_buffer(name)
        elements = _static_int(
            graph,
            graph.get_allocation_storage_size(buffer),
            "output allocation storage length",
        )
        item_size = torch.empty(
            (), device="meta", dtype=buffer.get_dtype()
        ).element_size()
    except (AttributeError, NotImplementedError, RuntimeError, TypeError) as error:
        raise CaptureError(
            f"Inductor GraphLowering output has no allocation extent: root={name}"
        ) from error
    return elements * item_size


def _lowered_output_views(
    records: tuple[_LoweredOutput, ...],
    roots: tuple[StorageRoot, ...],
) -> tuple[OutputView, ...]:
    root_id_by_name = {
        name: root.root_id
        for name, root in zip(
            dict.fromkeys(record.root_name for record in records),
            roots,
            strict=True,
        )
    }
    return tuple(
        OutputView(
            leaf_index=record.semantic_view.leaf_index,
            root_id=root_id_by_name[record.root_name],
            offset_bytes=record.offset_bytes,
            span_bytes=record.span_bytes,
            shape=record.shape,
            stride=record.stride,
            dtype=record.dtype,
            layout=record.semantic_view.layout,
        )
        for record in records
    )


def _make_storage_contract(
    roots: tuple[StorageRoot, ...],
    output_views: tuple[OutputView, ...],
    mutations: tuple[MutationBinding, ...],
) -> TaskStorageContract:
    identity = {
        "roots": [root.identity() for root in roots],
        "output_views": [view.identity() for view in output_views],
        "mutations": [mutation.identity() for mutation in mutations],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return TaskStorageContract(
        roots,
        output_views,
        mutations,
        hashlib.sha256(encoded.encode()).hexdigest(),
    )


def _visible_output_indices(graph: GraphModule) -> tuple[int, ...]:
    output = next(
        (node for node in graph.graph.nodes if node.op == "output"),
        None,
    )
    if output is None:
        raise CaptureError("optimized Inductor graph has no output node")
    raw_visible = output.meta.get("user_visible_output_idxs")
    if not isinstance(raw_visible, tuple | list) or any(
        not isinstance(index, int) or index < 0 for index in raw_visible
    ):
        raise CaptureError("Inductor did not publish callable-visible output indices")
    return tuple(raw_visible)


def _project_mutations(
    semantic_contract: TaskStorageContract,
    roots: tuple[StorageRoot, ...],
    output_views: tuple[OutputView, ...],
) -> tuple[MutationBinding, ...]:
    view_by_leaf = {view.leaf_index: view for view in output_views}
    root_by_id = {root.root_id: root for root in roots}
    for mutation in semantic_contract.mutations:
        leaf = mutation.replacement_output_leaf
        if leaf is None:
            continue
        view = view_by_leaf.get(leaf)
        if view is None:
            raise CaptureError(
                "Inductor removed a functional mutation replacement: "
                f"leaf={leaf}, target={mutation.argument_name}"
            )
        root = root_by_id[view.root_id]
        if (
            root.kind is StorageRootKind.INPUT
            and root.source_input != mutation.input_position
        ):
            raise CaptureError(
                "functional mutation replacement aliases another input: "
                f"leaf={leaf}, expected={mutation.input_position}, "
                f"actual={root.source_input}"
            )
    return semantic_contract.mutations


def _static_int(graph: GraphLowering, value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            sizevars: Any = graph.sizevars
            hinted = sizevars.size_hint(value)
        except BaseException as exc:
            raise CaptureError(f"Inductor {field} is not fixed-shape: {value}") from exc
        return int(hinted)


def _span_bytes(shape: tuple[int, ...], stride: tuple[int, ...], item_size: int) -> int:
    if not shape or any(extent == 0 for extent in shape):
        return 0 if shape and any(extent == 0 for extent in shape) else item_size
    last_element = sum(
        (extent - 1) * step for extent, step in zip(shape, stride, strict=True)
    )
    return (1 + last_element) * item_size


def _copy_root(root: StorageRoot, root_id: int) -> StorageRoot:
    return StorageRoot(
        root_id,
        root.kind,
        root.source_input,
        root.producer_node,
        root.producer_target,
        root.producer_result,
        root.minimum_span_bytes,
    )


def _copy_view(view: OutputView, leaf_index: int, root_id: int) -> OutputView:
    return OutputView(
        leaf_index,
        root_id,
        view.offset_bytes,
        view.span_bytes,
        view.shape,
        view.stride,
        view.dtype,
        view.layout,
    )


def _validate_value_abi(
    semantic: TaskStorageContract,
    executable: TaskStorageContract,
) -> None:
    """Require Inductor to preserve output values while allowing new aliases."""

    semantic_views = {view.leaf_index: view for view in semantic.output_views}
    executable_views = {view.leaf_index: view for view in executable.output_views}
    if semantic_views.keys() != executable_views.keys():
        raise CaptureError(
            "Inductor changed the tensor output leaves of the task ABI: "
            f"semantic={sorted(semantic_views)}, "
            f"executable={sorted(executable_views)}"
        )
    for leaf_index, semantic_view in semantic_views.items():
        executable_view = executable_views[leaf_index]
        semantic_geometry = (
            semantic_view.shape,
            semantic_view.stride,
            semantic_view.dtype,
            semantic_view.layout,
            semantic_view.span_bytes,
        )
        executable_geometry = (
            executable_view.shape,
            executable_view.stride,
            executable_view.dtype,
            executable_view.layout,
            executable_view.span_bytes,
        )
        same_significant_strides = all(
            extent <= 1 or semantic_stride == executable_stride
            for extent, semantic_stride, executable_stride in zip(
                semantic_view.shape,
                semantic_view.stride,
                executable_view.stride,
                strict=True,
            )
        )
        if (
            semantic_view.shape != executable_view.shape
            or semantic_view.dtype != executable_view.dtype
            or semantic_view.layout != executable_view.layout
            or semantic_view.span_bytes != executable_view.span_bytes
            or not same_significant_strides
        ):
            raise CaptureError(
                "Inductor changed task output geometry: "
                f"leaf={leaf_index}, semantic={semantic_geometry}, "
                f"executable={executable_geometry}"
            )
    semantic_mutations = {
        (item.input_position, item.replacement_output_leaf)
        for item in semantic.mutations
    }
    executable_mutations = {
        (item.input_position, item.replacement_output_leaf)
        for item in executable.mutations
    }
    if semantic_mutations != executable_mutations:
        raise CaptureError(
            "Inductor changed the task mutation ABI: "
            f"semantic={sorted(semantic_mutations)}, "
            f"executable={sorted(executable_mutations)}"
        )


__all__ = [
    "ExecutableRootAllocation",
    "ExecutableTaskManifest",
    "InductorCompilation",
    "compile_explicit_inductor_task",
    "compile_inductor_task",
]
