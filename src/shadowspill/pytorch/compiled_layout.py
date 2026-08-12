"""Physical output layout observed for one isolated compiled task ABI.

The semantic contract is authoritative for ownership and aliasing.  This
module accepts only physical observations that satisfy that contract; it can
never merge or split semantic roots to accommodate allocator behavior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .contracts import CaptureError
from .output_contract import StorageRootKind, TaskStorageContract
from .profiling import TaskAllocationEvent, TaskAllocationOperation, TaskMeasurement


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
    """Validated physical ABI for one semantic task contract."""

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
            raise ValueError("compiled root layouts must have dense IDs")
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


def reconcile_compiled_task_layout(
    contract: TaskStorageContract,
    measurement: TaskMeasurement,
) -> CompiledTaskLayout:
    """Validate allocator observations against an immutable semantic contract."""

    allocation_by_ordinal = {
        event.allocation_ordinal: event
        for event in measurement.allocation_trace
        if event.operation is TaskAllocationOperation.ALLOCATE
    }
    physical_by_leaf: dict[int, tuple[TaskAllocationEvent, int]] = {}
    for event in allocation_by_ordinal.values():
        for leaf_index, offset_bytes in zip(
            event.output_leaf_indices,
            event.output_view_offsets,
            strict=True,
        ):
            if leaf_index in physical_by_leaf:
                raise CaptureError(
                    f"compiled output leaf {leaf_index} has multiple allocations"
                )
            physical_by_leaf[leaf_index] = (event, offset_bytes)

    input_by_leaf = {
        item.output_leaf_index: item for item in measurement.output_input_bindings
    }

    semantic_views = {view.leaf_index: view for view in contract.output_views}
    unexpected = sorted(
        (set(physical_by_leaf) | set(input_by_leaf)) - set(semantic_views)
    )
    if unexpected:
        raise CaptureError(
            f"compiled layout reports unknown output leaves {unexpected}"
        )

    root_layouts: list[CompiledRootLayout] = []
    output_layouts: list[CompiledOutputView] = []
    claimed_allocations: dict[int, int] = {}
    views_by_root = {
        root.root_id: tuple(
            view for view in contract.output_views if view.root_id == root.root_id
        )
        for root in contract.roots
    }
    for root in contract.roots:
        views = views_by_root[root.root_id]
        if root.kind is StorageRootKind.INPUT:
            if any(view.leaf_index in physical_by_leaf for view in views):
                raise CaptureError(
                    f"input root {root.root_id} unexpectedly allocated an output"
                )
            nonempty = tuple(view for view in views if view.span_bytes > 0)
            missing = sorted(
                view.leaf_index
                for view in nonempty
                if view.leaf_index not in input_by_leaf
            )
            if missing:
                raise CaptureError(
                    f"input root {root.root_id} has unobserved output leaves "
                    f"{missing}"
                )
            if any(
                input_by_leaf[view.leaf_index].input_position != root.source_input
                for view in nonempty
                if view.leaf_index in input_by_leaf
            ):
                raise CaptureError(
                    f"input root {root.root_id} changed its compiled input storage"
                )
            root_layouts.append(CompiledRootLayout(root.root_id, None, 0, 0))
            output_layouts.extend(
                CompiledOutputView(
                    view.leaf_index,
                    root.root_id,
                    None,
                    (
                        input_by_leaf[view.leaf_index].output_offset_bytes
                        if view.leaf_index in input_by_leaf
                        else view.offset_bytes
                    ),
                )
                for view in views
            )
            continue

        nonempty = tuple(view for view in views if view.span_bytes > 0)
        bindings = tuple(
            physical_by_leaf[view.leaf_index]
            for view in nonempty
            if view.leaf_index in physical_by_leaf
        )
        if not nonempty and not bindings:
            root_layouts.append(CompiledRootLayout(root.root_id, None, 0, 0))
            output_layouts.extend(
                CompiledOutputView(view.leaf_index, root.root_id, None, 0)
                for view in views
            )
            continue
        donated = tuple(
            input_by_leaf[view.leaf_index]
            for view in nonempty
            if view.leaf_index in input_by_leaf
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
                if view.leaf_index not in physical_by_leaf
            )
            raise CaptureError(
                f"fresh root {root.root_id} has unobserved output leaves {missing}"
            )
        ordinals = {event.allocation_ordinal for event, _offset in bindings}
        if len(ordinals) != 1:
            leaf_bindings = [
                {
                    "leaf_index": view.leaf_index,
                    "semantic_offset_bytes": view.offset_bytes,
                    "span_bytes": view.span_bytes,
                    "allocation_ordinal": event.allocation_ordinal,
                    "allocation_requested_bytes": event.requested_bytes,
                    "allocation_charged_bytes": event.charged_bytes,
                    "physical_offset_bytes": offset,
                }
                for view, (event, offset) in zip(nonempty, bindings, strict=True)
            ]
            raise CaptureError(
                f"fresh root {root.root_id} spans compiled allocations "
                f"{sorted(ordinals)}; root={root.identity()!r}; "
                f"leaf_bindings={leaf_bindings!r}"
            )
        ordinal = next(iter(ordinals))
        prior_root = claimed_allocations.setdefault(ordinal, root.root_id)
        if prior_root != root.root_id:
            raise CaptureError(
                f"distinct fresh roots {prior_root} and {root.root_id} share "
                f"compiled allocation {ordinal}"
            )
        event = allocation_by_ordinal[ordinal]
        offsets = {
            view.leaf_index: physical_by_leaf[view.leaf_index][1]
            for view in nonempty
        }
        semantic_origin = min(view.offset_bytes for view in nonempty)
        physical_origin = min(offsets.values())
        for view in nonempty:
            actual = offsets[view.leaf_index]
            if actual - physical_origin != view.offset_bytes - semantic_origin:
                raise CaptureError(
                    f"compiled view {view.leaf_index} changes the relative layout "
                    f"of semantic root {root.root_id}"
                )
            if actual + view.span_bytes > event.requested_bytes:
                raise CaptureError(
                    f"compiled view {view.leaf_index} exceeds allocation {ordinal}"
                )
        root_layouts.append(
            CompiledRootLayout(
                root.root_id,
                ordinal,
                event.requested_bytes,
                event.charged_bytes,
            )
        )
        for view in views:
            if view.leaf_index in physical_by_leaf:
                offset = physical_by_leaf[view.leaf_index][1]
                allocation_ordinal: int | None = ordinal
            elif view.span_bytes == 0:
                offset = physical_origin + view.offset_bytes - semantic_origin
                allocation_ordinal = ordinal
            else:
                raise AssertionError("nonempty view binding was already validated")
            output_layouts.append(
                CompiledOutputView(
                    view.leaf_index,
                    root.root_id,
                    allocation_ordinal,
                    offset,
                )
            )

    identity = {
        "contract_digest": contract.compatibility_digest,
        "roots": [item.identity() for item in root_layouts],
        "output_views": [item.identity() for item in output_layouts],
        "allocation_trace": [
            event.to_dict() for event in measurement.allocation_trace
        ],
        "anonymous_workspace_high_water": measurement.workspace_charged_bytes,
        "anonymous_workspace_extents": list(measurement.workspace_extent_bytes),
        "persistent_provider_extents": list(measurement.persistent_extent_bytes),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return CompiledTaskLayout(
        contract_digest=contract.compatibility_digest,
        roots=tuple(root_layouts),
        output_views=tuple(sorted(output_layouts, key=lambda item: item.leaf_index)),
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
