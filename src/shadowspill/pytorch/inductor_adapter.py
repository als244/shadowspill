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
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._pytree import tree_flatten, tree_unflatten

from ._inductor_manifest_cache import (
    CachedTaskManifest,
    load_task_manifest,
    store_task_manifest,
)
from .contracts import CaptureError
from .output_contract import (
    MutationBinding,
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
    capture_task_storage_contract,
)


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
            raise ValueError("executable root allocations must have dense root IDs")
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


def _manifest_inner_compile(
    semantic_contract: TaskStorageContract,
    manifests: list[ExecutableTaskManifest],
    *,
    canonicalize_input_aliases: bool = False,
    phase_timings: dict[str, int] | None = None,
) -> Callable[..., object]:
    """Build the common Inductor callback that publishes one physical ABI."""

    inner_backend: Any = compile_fx_inner

    def record(name: str, started: int) -> int:
        duration = time.perf_counter_ns() - started
        if phase_timings is not None:
            phase_timings[name] = phase_timings.get(name, 0) + duration
        return duration

    def inner_compile(
        optimized_graph: GraphModule,
        optimized_inputs: Sequence[object],
        **options: Any,
    ) -> object:
        alias_started = time.perf_counter_ns()
        if canonicalize_input_aliases:
            _canonicalize_input_alias_outputs(
                optimized_graph,
                semantic_contract,
                tuple(optimized_inputs),
            )
        alias_ns = record("shadowspill_input_alias_normalization", alias_started)

        contract_started = time.perf_counter_ns()
        inner_contract = capture_task_storage_contract(
            optimized_graph,
            tuple(optimized_inputs),
        )
        optimized_contract = _project_callable_contract(
            optimized_graph,
            inner_contract,
            semantic_contract,
        )
        _validate_value_abi(semantic_contract, optimized_contract)
        tracing_context = TracingContext.try_get()
        if (
            tracing_context is not None
            and tracing_context.fake_mode is not None
            and tracing_context.fake_mode.shape_env is None
        ):
            tracing_context.fake_mode.shape_env = ShapeEnv()
        optimized_contract_ns = record(
            "shadowspill_optimized_contract", contract_started
        )

        inductor_started = time.perf_counter_ns()
        with _GRAPH_LOWERING_CAPTURE_LOCK:
            compiled, graph_lowerings = run_and_get_graph_lowering(
                lambda: inner_backend(
                    optimized_graph,
                    optimized_inputs,
                    **options,
                )
            )
        record("torch_inductor_core", inductor_started)

        sidecar_started = time.perf_counter_ns()
        cache_key = _fx_graph_cache_key(compiled)
        record("shadowspill_manifest_sidecar", sidecar_started)
        if len(graph_lowerings) == 1:
            executable_started = time.perf_counter_ns()
            executable = _graph_lowering_contract(
                graph_lowerings[0],
                optimized_graph,
                inner_contract,
                semantic_contract,
            )
            executable_contract_ns = record(
                "shadowspill_executable_contract", executable_started
            )
            manifest_started = time.perf_counter_ns()
            manifest = _make_manifest(
                semantic_contract,
                optimized_contract,
                executable.storage_contract,
                executable.root_allocations,
                capture_ns=(alias_ns + optimized_contract_ns + executable_contract_ns),
            )
            manifests.append(manifest)
            record("shadowspill_manifest_assembly", manifest_started)
            if cache_key is not None:
                sidecar_started = time.perf_counter_ns()
                _store_cached_manifest(cache_key, manifest)
                record("shadowspill_manifest_sidecar", sidecar_started)
            return compiled
        if len(graph_lowerings) > 1:
            raise CaptureError(
                "Inductor exposed multiple GraphLowering results: "
                f"observed={len(graph_lowerings)}"
            )
        if cache_key is not None:
            sidecar_started = time.perf_counter_ns()
            cached = _load_cached_manifest(
                cache_key,
                semantic_contract,
                optimized_contract=optimized_contract,
                capture_ns=alias_ns + optimized_contract_ns,
            )
            record("shadowspill_manifest_sidecar", sidecar_started)
            if cached is not None:
                manifests.append(cached)
        return compiled

    return inner_compile


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
        raise CaptureError(f"Inductor task compilation failed: {exc}") from exc
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
                raise CaptureError(
                    f"Inductor task manifest regeneration failed: {exc}"
                ) from exc
            if len(manifests) == 1 and cache_key is not None:
                _store_cached_manifest(cache_key, manifests[0])
    if len(manifests) != 1:
        raise CaptureError(
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
    """Compile an already-explicit task without a second AOTAutograd pass.

    This is the differential candidate for ShadowSpill's production compiler
    path.  It deliberately uses the same decomposition table and Inductor
    forward compiler as ``compile_fx``, but normalizes the explicit FX task
    directly and therefore never asks AOTAutograd to rediscover a forward or
    backward program that ShadowSpill already owns.
    """

    phase_timings: dict[str, int] = {}

    def record(name: str, started: int) -> int:
        duration = time.perf_counter_ns() - started
        phase_timings[name] = phase_timings.get(name, 0) + duration
        return duration

    input_started = time.perf_counter_ns()
    source_graph = graph_module
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
    tracing_context = TracingContext(fake_mode)
    record("shadowspill_compiler_input_setup", input_started)

    normalization_started = time.perf_counter_ns()
    try:
        with V.set_fake_mode(fake_mode), tracing(tracing_context):
            normalized = make_fx(
                source_graph,
                decomposition_table=select_decomp_table(),
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(*fake_inputs)
    except BaseException as exc:
        raise CaptureError(
            f"explicit Inductor task normalization failed: {exc}"
        ) from exc
    record("torch_decomposition_normalization", normalization_started)

    output_started = time.perf_counter_ns()
    output_node = next(node for node in normalized.graph.nodes if node.op == "output")
    output_leaves, output_spec = tree_flatten(output_node.args[0])
    output_node.args = (tuple(output_leaves),)
    normalized.graph.lint()
    normalized.recompile()
    record("shadowspill_output_abi_normalization", output_started)

    manifests: list[ExecutableTaskManifest] = []
    inner_compile = _manifest_inner_compile(
        semantic_contract,
        manifests,
        canonicalize_input_aliases=True,
        phase_timings=phase_timings,
    )
    config_started = time.perf_counter_ns()
    compiler_config = create_compiler_config_extra(normalized)
    record("torch_compiler_configuration", config_started)

    def invoke_compiler() -> object:
        nested_before = sum(phase_timings.values())
        invocation_started = time.perf_counter_ns()
        try:
            with V.set_fake_mode(fake_mode), tracing(TracingContext(fake_mode)):
                compiler = cast(Callable[..., object], compile_fx_forward)
                return compiler(
                    normalized,
                    fake_inputs,
                    num_orig_model_outputs=len(output_leaves),
                    num_example_inputs=len(fake_inputs),
                    compiler_config_extra=compiler_config,
                    inner_compile=inner_compile,
                    is_inference=True,
                )
        finally:
            elapsed = time.perf_counter_ns() - invocation_started
            nested_elapsed = sum(phase_timings.values()) - nested_before
            phase_timings["torch_compile_fx_forward_orchestration"] = phase_timings.get(
                "torch_compile_fx_forward_orchestration", 0
            ) + max(0, elapsed - nested_elapsed)

    try:
        compiled = invoke_compiler()
    except BaseException as exc:
        raise CaptureError(f"explicit Inductor task compilation failed: {exc}") from exc

    if not manifests:
        sidecar_started = time.perf_counter_ns()
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
        record("shadowspill_manifest_sidecar", sidecar_started)
        if not manifests:
            try:
                with inductor_config.patch({"force_disable_caches": True}):
                    compiled = invoke_compiler()
            except BaseException as exc:
                raise CaptureError(
                    f"explicit Inductor manifest regeneration failed: {exc}"
                ) from exc
    if len(manifests) != 1:
        raise CaptureError(
            "explicit Inductor compilation did not expose one root graph: "
            f"observed={len(manifests)}"
        )

    wrapper_started = time.perf_counter_ns()
    compiled_callable = cast(
        Callable[[list[object]], Sequence[object]],
        compiled,
    )

    def unboxed(*arguments: object) -> object:
        values = compiled_callable(list(arguments))
        return tree_unflatten(list(values), output_spec)

    record("shadowspill_callable_wrapper", wrapper_started)
    return InductorCompilation(
        unboxed,
        manifests[0],
        _ordered_compilation_timings(phase_timings),
    )


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
        source_position = root.source_input
        if source_position is None:
            raise AssertionError("input storage root omitted its source position")
        source = example_inputs[source_position]
        if not isinstance(source, torch.Tensor):
            raise CaptureError("compiled input alias refers to a non-tensor argument")
        if view.dtype != str(source.dtype):
            raise CaptureError(
                "direct compilation cannot express a dtype-changing input view: "
                f"leaf={leaf_index}, input={source_position}, "
                f"source={source.dtype}, output={view.dtype}"
            )
        itemsize = source.element_size()
        if view.offset_bytes % itemsize:
            raise CaptureError(
                "input-alias output offset is not element aligned: "
                f"leaf={leaf_index}, bytes={view.offset_bytes}, itemsize={itemsize}"
            )
        placeholder = placeholders[source_position]
        source_offset_bytes = int(source.storage_offset()) * itemsize
        if (
            view.shape == tuple(int(value) for value in source.shape)
            and view.stride == tuple(int(value) for value in source.stride())
            and view.offset_bytes == source_offset_bytes
        ):
            replacement = placeholder
        else:
            with graph_module.graph.inserting_before(output_node):
                replacement = graph_module.graph.call_function(
                    torch.ops.aten.as_strided.default,
                    args=(
                        placeholder,
                        view.shape,
                        view.stride,
                        view.offset_bytes // itemsize,
                    ),
                )
        if replacement is not leaf:
            leaves[leaf_index] = replacement
            changed = True
    if changed:
        output_node.args = (tree_unflatten(leaves, spec),)
        graph_module.graph.lint()
        graph_module.recompile()


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
    except (CaptureError, ValueError):
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
    dense_root_id = {
        original: dense for dense, original in enumerate(selected_root_ids)
    }
    roots = tuple(
        _copy_root(inner_root_by_id[original], dense_root_id[original])
        for original in selected_root_ids
    )
    output_views = tuple(
        _copy_view(executable, semantic.leaf_index, dense_root_id[executable.root_id])
        for executable, semantic in zip(visible_views, semantic_views, strict=True)
    )
    view_by_leaf = {view.leaf_index: view for view in output_views}
    root_by_id = {root.root_id: root for root in roots}
    mutations: list[MutationBinding] = []
    for mutation in semantic_contract.mutations:
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
        mutations.append(mutation)
    identity = {
        "roots": [root.identity() for root in roots],
        "output_views": [view.identity() for view in output_views],
        "mutations": [mutation.identity() for mutation in mutations],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return TaskStorageContract(
        roots,
        output_views,
        tuple(mutations),
        hashlib.sha256(encoded.encode()).hexdigest(),
    )


def _graph_lowering_contract(
    graph: GraphLowering,
    optimized_graph: GraphModule,
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> _GraphLoweringManifest:
    """Project Inductor's final returned buffers into a storage contract.

    Post-gradient FX still describes value provenance. During lowering and
    fusion Inductor may realize disjoint views of one FX value into separate
    returned buffers. ``GraphLowering.graph_outputs`` is the last structured
    representation that names the storage roots used by the generated wrapper.
    """

    visible = _visible_output_indices(optimized_graph)
    if graph.graph_outputs is None:
        raise CaptureError("Inductor GraphLowering omitted task outputs")
    if any(index >= len(graph.graph_outputs) for index in visible):
        raise CaptureError(
            "Inductor callable-visible output index exceeds GraphLowering ABI"
        )
    inner_view_by_leaf = {view.leaf_index: view for view in inner_contract.output_views}
    selected_outputs = tuple(
        graph.graph_outputs[index] for index in visible if index in inner_view_by_leaf
    )
    semantic_views = tuple(
        sorted(semantic_contract.output_views, key=lambda view: view.leaf_index)
    )
    if len(selected_outputs) != len(semantic_views):
        raise CaptureError(
            "GraphLowering callable-visible tensor output count changed: "
            f"semantic={len(semantic_views)}, "
            f"executable={len(selected_outputs)}"
        )

    placeholders = tuple(
        node for node in optimized_graph.graph.nodes if node.op == "placeholder"
    )
    input_position_by_name = {
        node.name: index
        for index, node in enumerate(placeholders)
        if node.name in graph.graph_inputs
    }
    optimized_views = tuple(
        inner_view_by_leaf[index] for index in visible if index in inner_view_by_leaf
    )
    optimized_root_by_id = {root.root_id: root for root in inner_contract.roots}

    records: list[_LoweredOutput] = []
    for semantic_view, optimized_view, output in zip(
        semantic_views,
        optimized_views,
        selected_outputs,
        strict=True,
    ):
        if not output.has_tensor_output():
            raise CaptureError(
                "GraphLowering replaced a tensor output with a non-tensor value: "
                f"leaf={semantic_view.leaf_index}"
            )
        try:
            root_name = str(output.get_name())
            layout = output.get_layout()
            dtype = output.get_dtype()
            shape = tuple(
                _static_int(graph, value, "output shape") for value in output.get_size()
            )
            stride = tuple(
                _static_int(graph, value, "output stride")
                for value in output.get_stride()
            )
            offset_elements = _static_int(graph, layout.offset, "output offset")
        except (AttributeError, NotImplementedError, TypeError) as exc:
            raise CaptureError(
                "Inductor GraphLowering output has no concrete strided layout: "
                f"leaf={semantic_view.leaf_index}, type={type(output).__name__}"
            ) from exc
        if offset_elements < 0 or any(value < 0 for value in (*shape, *stride)):
            raise CaptureError(
                "Inductor GraphLowering produced a negative output geometry: "
                f"leaf={semantic_view.leaf_index}"
            )
        item_size = torch.empty((), device="meta", dtype=dtype).element_size()
        offset_bytes = offset_elements * item_size
        span_bytes = _span_bytes(shape, stride, item_size)
        expected_geometry = (
            semantic_view.shape,
            semantic_view.stride,
            semantic_view.dtype,
            semantic_view.layout,
        )
        same_significant_strides = all(
            extent <= 1 or actual == expected
            for extent, actual, expected in zip(
                shape,
                stride,
                semantic_view.stride,
                strict=True,
            )
        )
        if (
            shape != semantic_view.shape
            or str(dtype) != semantic_view.dtype
            or str(torch.strided) != semantic_view.layout
            or not same_significant_strides
        ):
            actual_geometry = (shape, stride, str(dtype), str(torch.strided))
            raise CaptureError(
                "GraphLowering changed task output geometry: "
                f"leaf={semantic_view.leaf_index}, expected={expected_geometry}, "
                f"actual={actual_geometry}"
            )
        records.append(
            _LoweredOutput(
                semantic_view=semantic_view,
                optimized_view=optimized_view,
                provenance=optimized_root_by_id[optimized_view.root_id],
                root_name=root_name,
                offset_bytes=offset_bytes,
                span_bytes=span_bytes,
                shape=shape,
                stride=stride,
                dtype=str(dtype),
            )
        )

    root_order = tuple(dict.fromkeys(record.root_name for record in records))
    root_id_by_name = {name: index for index, name in enumerate(root_order)}
    roots: list[StorageRoot] = []
    root_allocations: list[ExecutableRootAllocation] = []
    for name in root_order:
        members = tuple(record for record in records if record.root_name == name)
        provenance = members[0].provenance
        minimum_span = max(
            record.offset_bytes + record.span_bytes for record in members
        )
        source_input = input_position_by_name.get(name)
        if source_input is not None:
            roots.append(
                StorageRoot(
                    root_id_by_name[name],
                    StorageRootKind.INPUT,
                    source_input,
                    None,
                    None,
                    None,
                    minimum_span,
                )
            )
            root_allocations.append(ExecutableRootAllocation(root_id_by_name[name], 0))
            continue
        try:
            buffer: Any = graph.get_buffer(name)
            allocation_elements = _static_int(
                graph,
                graph.get_allocation_storage_size(buffer),
                "output allocation storage length",
            )
            allocation_item_size = torch.empty(
                (), device="meta", dtype=buffer.get_dtype()
            ).element_size()
        except (AttributeError, NotImplementedError, RuntimeError, TypeError) as exc:
            raise CaptureError(
                f"Inductor GraphLowering output has no allocation extent: root={name}"
            ) from exc
        allocation_bytes = allocation_elements * allocation_item_size
        if allocation_bytes < minimum_span:
            raise CaptureError(
                "Inductor output allocation is smaller than its returned views: "
                f"root={name}, allocation={allocation_bytes}, "
                f"minimum_span={minimum_span}"
            )
        roots.append(
            StorageRoot(
                root_id_by_name[name],
                StorageRootKind.FRESH,
                None,
                provenance.producer_node or f"inductor_{name}",
                provenance.producer_target or "inductor.output_buffer",
                provenance.producer_result or 0,
                minimum_span,
            )
        )
        root_allocations.append(
            ExecutableRootAllocation(root_id_by_name[name], allocation_bytes)
        )

    output_views = tuple(
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
    mutations = _project_mutations(semantic_contract, tuple(roots), output_views)
    identity = {
        "roots": [root.identity() for root in roots],
        "output_views": [view.identity() for view in output_views],
        "mutations": [mutation.identity() for mutation in mutations],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return _GraphLoweringManifest(
        TaskStorageContract(
            tuple(roots),
            output_views,
            mutations,
            hashlib.sha256(encoded.encode()).hexdigest(),
        ),
        tuple(root_allocations),
    )


def _visible_output_indices(graph: GraphModule) -> tuple[int, ...]:
    output = next(
        (node for node in graph.graph.nodes if node.op == "output"),
        None,
    )
    if output is None:
        raise CaptureError("optimized Inductor graph has no output node")
    raw_visible = output.meta.get("user_visible_output_idxs")
    if not isinstance(raw_visible, (tuple, list)) or any(
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
