"""Physical output layout observed for one isolated compiled task contract.

The semantic contract is authoritative for ownership and aliasing.  This
module accepts only physical observations that satisfy that contract; it can
never merge or split semantic roots to accommodate allocator behavior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from shadowspill.pytorch.capture.storage import (
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)
from shadowspill.pytorch.compilation.inductor import ExecutableRootAllocation
from shadowspill.pytorch.contracts import CaptureError
from shadowspill.pytorch.profiling.records import (
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
    TaskOutputInputBinding,
)


@dataclass(frozen=True, slots=True)
class CompiledRootLayout:
    """Physical allocation serving one semantic root for this executable."""

    root_id: int
    allocation_ordinal: int | None
    requested_bytes: int
    charged_bytes: int

    def __post_init__(self) -> None:
        if self.root_id < 0 or self.requested_bytes < 0 or self.charged_bytes < 0:
            raise ValueError("compiled root layout fields must be non-negative")
        if self.allocation_ordinal is None:
            if self.requested_bytes or self.charged_bytes:
                raise ValueError("unallocated root cannot have a physical extent")
        elif self.allocation_ordinal < 0 or self.charged_bytes < self.requested_bytes:
            raise ValueError("compiled root allocation is invalid")

    def identity(self) -> dict[str, object]:
        return {
            "root_id": self.root_id,
            "allocation_ordinal": self.allocation_ordinal,
            "requested_bytes": self.requested_bytes,
            "charged_bytes": self.charged_bytes,
        }


@dataclass(frozen=True, slots=True)
class CompiledOutputView:
    """Physical binding for one tensor leaf returned by the executable."""

    leaf_index: int
    root_id: int
    allocation_ordinal: int | None
    offset_bytes: int

    def __post_init__(self) -> None:
        if min(self.leaf_index, self.root_id, self.offset_bytes) < 0:
            raise ValueError("compiled output-view fields must be non-negative")
        if self.allocation_ordinal is not None and self.allocation_ordinal < 0:
            raise ValueError("compiled output allocation ordinal is invalid")

    def identity(self) -> dict[str, object]:
        return {
            "leaf_index": self.leaf_index,
            "root_id": self.root_id,
            "allocation_ordinal": self.allocation_ordinal,
            "offset_bytes": self.offset_bytes,
        }


@dataclass(frozen=True, slots=True)
class CompiledTaskLayout:
    """Validated physical contract for one semantic task contract."""

    contract_digest: str
    roots: tuple[CompiledRootLayout, ...]
    output_views: tuple[CompiledOutputView, ...]
    allocation_trace: tuple[TaskAllocationEvent, ...]
    anonymous_workspace_high_water: int
    anonymous_workspace_extents: tuple[int, ...]
    persistent_provider_extents: tuple[int, ...]
    compatibility_digest: str

    def __post_init__(self) -> None:
        if len(self.contract_digest) != 64 or len(self.compatibility_digest) != 64:
            raise ValueError("compiled task layout digests must be SHA-256")
        if tuple(root.root_id for root in self.roots) != tuple(range(len(self.roots))):
            raise ValueError("compiled root layouts must have contiguous indices")
        leaves = tuple(view.leaf_index for view in self.output_views)
        if len(set(leaves)) != len(leaves):
            raise ValueError("compiled output leaf is bound more than once")
        values = (
            self.anonymous_workspace_high_water,
            *self.anonymous_workspace_extents,
            *self.persistent_provider_extents,
        )
        if any(value < 0 for value in values):
            raise ValueError("compiled task physical extents must be non-negative")

    def identity(self) -> dict[str, object]:
        return {
            "contract_digest": self.contract_digest,
            "roots": [root.identity() for root in self.roots],
            "output_views": [view.identity() for view in self.output_views],
            "allocation_trace": [event.to_dict() for event in self.allocation_trace],
            "anonymous_workspace_high_water": self.anonymous_workspace_high_water,
            "anonymous_workspace_extents": list(self.anonymous_workspace_extents),
            "persistent_provider_extents": list(self.persistent_provider_extents),
        }

    def root(self, root_id: int) -> CompiledRootLayout:
        try:
            return self.roots[root_id]
        except IndexError as exc:
            raise CaptureError("compiled layout references an unknown root") from exc

    def to_json(self) -> str:
        """Return deterministic standalone diagnostic serialization."""

        return json.dumps(
            {
                "schema": "shadowspill.compiled_task_layout/v1",
                "compatibility_digest": self.compatibility_digest,
                **self.identity(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class _PhysicalObservations:
    compiler_extent: dict[int, int] | None
    allocation_by_ordinal: dict[int, TaskAllocationEvent]
    physical_by_leaf: dict[int, tuple[TaskAllocationEvent, int]]
    input_by_leaf: dict[int, TaskOutputInputBinding]


@dataclass(slots=True)
class _LayoutBuilder:
    roots: list[CompiledRootLayout]
    output_views: list[CompiledOutputView]
    claimed_allocations: dict[int, int]


@dataclass(frozen=True, slots=True)
class _FreshAllocation:
    ordinal: int
    event: TaskAllocationEvent
    requested_bytes: int
    semantic_origin: int
    physical_origin: int


def reconcile_compiled_task_layout(
    contract: TaskStorageContract,
    measurement: TaskMeasurement,
    *,
    root_allocations: tuple[ExecutableRootAllocation, ...] | None = None,
) -> CompiledTaskLayout:
    """Validate physical observations without changing semantic ownership."""

    observations = _index_physical_observations(contract, measurement, root_allocations)
    views_by_root = _views_by_root(contract)
    builder = _LayoutBuilder([], [], {})
    for root in contract.roots:
        views = views_by_root[root.root_id]
        if root.kind is StorageRootKind.INPUT:
            _reconcile_input_root(root, views, observations, builder)
        else:
            _reconcile_fresh_root(root, views, observations, builder)
    return _finish_compiled_layout(contract, measurement, builder)


def _index_physical_observations(
    contract: TaskStorageContract,
    measurement: TaskMeasurement,
    root_allocations: tuple[ExecutableRootAllocation, ...] | None,
) -> _PhysicalObservations:
    if root_allocations is not None and tuple(
        item.root_id for item in root_allocations
    ) != tuple(range(len(contract.roots))):
        raise CaptureError("compiled root allocations do not match the contract")
    allocation_by_ordinal = {
        event.allocation_ordinal: event
        for event in measurement.allocation_trace
        if event.operation is TaskAllocationOperation.ALLOCATE
    }
    physical_by_leaf = _index_physical_output_leaves(allocation_by_ordinal)
    input_by_leaf = {
        item.output_leaf_index: item for item in measurement.output_input_bindings
    }
    semantic_leaves = {view.leaf_index for view in contract.output_views}
    unexpected = sorted((set(physical_by_leaf) | set(input_by_leaf)) - semantic_leaves)
    if unexpected:
        raise CaptureError(
            f"compiled layout reports unknown output leaves {unexpected}"
        )
    return _PhysicalObservations(
        compiler_extent=(
            None
            if root_allocations is None
            else {item.root_id: item.requested_bytes for item in root_allocations}
        ),
        allocation_by_ordinal=allocation_by_ordinal,
        physical_by_leaf=physical_by_leaf,
        input_by_leaf=input_by_leaf,
    )


def _index_physical_output_leaves(
    allocations: dict[int, TaskAllocationEvent],
) -> dict[int, tuple[TaskAllocationEvent, int]]:
    leaves: dict[int, tuple[TaskAllocationEvent, int]] = {}
    for event in allocations.values():
        for leaf_index, offset_bytes in zip(
            event.output_leaf_indices,
            event.output_view_offsets,
            strict=True,
        ):
            if leaf_index in leaves:
                raise CaptureError(
                    f"compiled output leaf {leaf_index} has multiple allocations"
                )
            leaves[leaf_index] = event, offset_bytes
    return leaves


def _views_by_root(
    contract: TaskStorageContract,
) -> dict[int, tuple[OutputView, ...]]:
    return {
        root.root_id: tuple(
            view for view in contract.output_views if view.root_id == root.root_id
        )
        for root in contract.roots
    }


def _reconcile_input_root(
    root: StorageRoot,
    views: tuple[OutputView, ...],
    observations: _PhysicalObservations,
    builder: _LayoutBuilder,
) -> None:
    if any(view.leaf_index in observations.physical_by_leaf for view in views):
        raise CaptureError(
            f"input root {root.root_id} unexpectedly allocated an output"
        )
    nonempty = tuple(view for view in views if view.span_bytes > 0)
    missing = sorted(
        view.leaf_index
        for view in nonempty
        if view.leaf_index not in observations.input_by_leaf
    )
    if missing:
        raise CaptureError(
            f"input root {root.root_id} has unobserved output leaves {missing}"
        )
    if any(
        observations.input_by_leaf[view.leaf_index].input_position != root.source_input
        for view in nonempty
    ):
        raise CaptureError(
            f"input root {root.root_id} changed its compiled input storage"
        )
    builder.roots.append(CompiledRootLayout(root.root_id, None, 0, 0))
    builder.output_views.extend(
        CompiledOutputView(
            leaf_index=view.leaf_index,
            root_id=root.root_id,
            allocation_ordinal=None,
            offset_bytes=(
                observations.input_by_leaf[view.leaf_index].output_offset_bytes
                if view.leaf_index in observations.input_by_leaf
                else view.offset_bytes
            ),
        )
        for view in views
    )


def _reconcile_fresh_root(
    root: StorageRoot,
    views: tuple[OutputView, ...],
    observations: _PhysicalObservations,
    builder: _LayoutBuilder,
) -> None:
    nonempty = tuple(view for view in views if view.span_bytes > 0)
    bindings = tuple(
        observations.physical_by_leaf[view.leaf_index]
        for view in nonempty
        if view.leaf_index in observations.physical_by_leaf
    )
    if not nonempty and not bindings:
        _append_zero_root(root, views, builder)
        return
    _validate_fresh_bindings(root, nonempty, bindings, observations)
    allocation = _resolve_fresh_allocation(
        root, nonempty, bindings, observations, builder
    )
    _validate_fresh_view_offsets(root, nonempty, observations, allocation)
    builder.roots.append(
        CompiledRootLayout(
            root.root_id,
            allocation.ordinal,
            allocation.requested_bytes,
            allocation.event.charged_bytes,
        )
    )
    _append_fresh_output_views(root, views, observations, allocation, builder)


def _append_zero_root(
    root: StorageRoot,
    views: tuple[OutputView, ...],
    builder: _LayoutBuilder,
) -> None:
    builder.roots.append(CompiledRootLayout(root.root_id, None, 0, 0))
    builder.output_views.extend(
        CompiledOutputView(view.leaf_index, root.root_id, None, 0) for view in views
    )


def _validate_fresh_bindings(
    root: StorageRoot,
    nonempty: tuple[OutputView, ...],
    bindings: tuple[tuple[TaskAllocationEvent, int], ...],
    observations: _PhysicalObservations,
) -> None:
    donated = tuple(
        observations.input_by_leaf[view.leaf_index]
        for view in nonempty
        if view.leaf_index in observations.input_by_leaf
    )
    if bindings and donated:
        raise CaptureError(
            f"fresh root {root.root_id} mixes allocated and donated storage"
        )
    if donated:
        raise CaptureError(
            f"fresh executable root {root.root_id} unexpectedly aliases a "
            "task input; the offline Inductor contract is incomplete"
        )
    if len(bindings) != len(nonempty):
        missing = sorted(
            view.leaf_index
            for view in nonempty
            if view.leaf_index not in observations.physical_by_leaf
        )
        raise CaptureError(
            f"fresh root {root.root_id} has unobserved output leaves {missing}"
        )


def _resolve_fresh_allocation(
    root: StorageRoot,
    views: tuple[OutputView, ...],
    bindings: tuple[tuple[TaskAllocationEvent, int], ...],
    observations: _PhysicalObservations,
    builder: _LayoutBuilder,
) -> _FreshAllocation:
    ordinals = {event.allocation_ordinal for event, _offset in bindings}
    if len(ordinals) != 1:
        raise CaptureError(
            f"fresh root {root.root_id} spans compiled allocations "
            f"{sorted(ordinals)}; root={root.identity()!r}; "
            f"leaf_bindings={_leaf_binding_evidence(views, bindings)!r}"
        )
    ordinal = next(iter(ordinals))
    prior_root = builder.claimed_allocations.setdefault(ordinal, root.root_id)
    if prior_root != root.root_id:
        raise CaptureError(
            f"distinct fresh roots {prior_root} and {root.root_id} share "
            f"compiled allocation {ordinal}"
        )
    event = observations.allocation_by_ordinal[ordinal]
    requested = (
        event.requested_bytes
        if observations.compiler_extent is None
        else observations.compiler_extent[root.root_id]
    )
    if event.requested_bytes != requested:
        raise CaptureError(
            "allocator profile disagrees with Inductor output extent: "
            f"root={root.root_id}, compiler={requested}, "
            f"observed={event.requested_bytes}"
        )
    offsets = {
        view.leaf_index: observations.physical_by_leaf[view.leaf_index][1]
        for view in views
    }
    return _FreshAllocation(
        ordinal=ordinal,
        event=event,
        requested_bytes=requested,
        semantic_origin=min(view.offset_bytes for view in views),
        physical_origin=min(offsets.values()),
    )


def _leaf_binding_evidence(
    views: tuple[OutputView, ...],
    bindings: tuple[tuple[TaskAllocationEvent, int], ...],
) -> list[dict[str, int]]:
    return [
        {
            "leaf_index": view.leaf_index,
            "semantic_offset_bytes": view.offset_bytes,
            "span_bytes": view.span_bytes,
            "allocation_ordinal": event.allocation_ordinal,
            "allocation_requested_bytes": event.requested_bytes,
            "allocation_charged_bytes": event.charged_bytes,
            "physical_offset_bytes": offset,
        }
        for view, (event, offset) in zip(views, bindings, strict=True)
    ]


def _validate_fresh_view_offsets(
    root: StorageRoot,
    views: tuple[OutputView, ...],
    observations: _PhysicalObservations,
    allocation: _FreshAllocation,
) -> None:
    for view in views:
        actual = observations.physical_by_leaf[view.leaf_index][1]
        if (
            actual - allocation.physical_origin
            != view.offset_bytes - allocation.semantic_origin
        ):
            raise CaptureError(
                f"compiled view {view.leaf_index} changes the relative layout "
                f"of semantic root {root.root_id}"
            )
        if actual + view.span_bytes > allocation.event.requested_bytes:
            raise CaptureError(
                f"compiled view {view.leaf_index} exceeds allocation "
                f"{allocation.ordinal}"
            )


def _append_fresh_output_views(
    root: StorageRoot,
    views: tuple[OutputView, ...],
    observations: _PhysicalObservations,
    allocation: _FreshAllocation,
    builder: _LayoutBuilder,
) -> None:
    for view in views:
        if view.leaf_index in observations.physical_by_leaf:
            offset = observations.physical_by_leaf[view.leaf_index][1]
        elif view.span_bytes == 0:
            offset = (
                allocation.physical_origin
                + view.offset_bytes
                - allocation.semantic_origin
            )
        else:
            raise AssertionError("nonempty view binding was already validated")
        builder.output_views.append(
            CompiledOutputView(
                view.leaf_index,
                root.root_id,
                allocation.ordinal,
                offset,
            )
        )


def _finish_compiled_layout(
    contract: TaskStorageContract,
    measurement: TaskMeasurement,
    builder: _LayoutBuilder,
) -> CompiledTaskLayout:
    output_views = tuple(sorted(builder.output_views, key=lambda item: item.leaf_index))
    identity = {
        "contract_digest": contract.compatibility_digest,
        "roots": [item.identity() for item in builder.roots],
        "output_views": [item.identity() for item in output_views],
        "allocation_trace": [event.to_dict() for event in measurement.allocation_trace],
        "anonymous_workspace_high_water": measurement.workspace_charged_bytes,
        "anonymous_workspace_extents": list(measurement.workspace_extent_bytes),
        "persistent_provider_extents": list(measurement.persistent_extent_bytes),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return CompiledTaskLayout(
        contract_digest=contract.compatibility_digest,
        roots=tuple(builder.roots),
        output_views=output_views,
        allocation_trace=measurement.allocation_trace,
        anonymous_workspace_high_water=measurement.workspace_charged_bytes,
        anonymous_workspace_extents=measurement.workspace_extent_bytes,
        persistent_provider_extents=measurement.persistent_extent_bytes,
        compatibility_digest=hashlib.sha256(encoded.encode()).hexdigest(),
    )


def replacement_transition_bytes(
    contract: TaskStorageContract,
    layout: CompiledTaskLayout,
) -> int:
    """Return the exact old/new overlap required by functional mutations.

    Export represents some mutations as fresh returned allocations.  The old
    object generation must remain live until the task-completion fence while
    the returned allocation becomes the next generation.  Count each fresh
    physical root once even when several returned views name it.
    """

    if layout.contract_digest != contract.compatibility_digest:
        raise CaptureError("compiled task layout belongs to another contract")
    root_by_leaf = {view.leaf_index: view.root_id for view in contract.output_views}
    replacement_roots: set[int] = set()
    for mutation in contract.mutations:
        leaf_index = mutation.replacement_output_leaf
        if leaf_index is None:
            continue
        try:
            root_id = root_by_leaf[leaf_index]
            root = contract.roots[root_id]
        except (KeyError, IndexError) as exc:
            raise CaptureError("functional mutation has no output root") from exc
        if root.kind is StorageRootKind.INPUT:
            if root.source_input != mutation.input_position:
                raise CaptureError(
                    "functional mutation replacement aliases another task input"
                )
            continue
        replacement_roots.add(root_id)
    return sum(layout.root(root_id).charged_bytes for root_id in replacement_roots)


__all__ = [
    "CompiledOutputView",
    "CompiledRootLayout",
    "CompiledTaskLayout",
    "reconcile_compiled_task_layout",
    "replacement_transition_bytes",
]
