"""Semantic saved-value accounting for AOT forward/backward alternatives."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.pytorch.capture.artifacts import AotGraphPair
from shadowspill.pytorch.capture.storage import (
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)


@dataclass(frozen=True, slots=True)
class SavedValueFootprint:
    """Classify storage roots returned only to feed the paired backward graph.

    AOT may return task inputs and views of public outputs as saved values.
    Those leaves are backward arguments, but they do not allocate a new
    activation.  Only ``internal_root_ids`` are fresh, non-public roots whose
    lifetime is introduced by the selected graph-pair alternative.
    """

    public_output_count: int
    saved_value_count: int
    input_root_ids: tuple[int, ...]
    boundary_root_ids: tuple[int, ...]
    internal_root_ids: tuple[int, ...]
    input_minimum_bytes: int
    boundary_minimum_bytes: int
    internal_minimum_bytes: int


def saved_value_footprint(
    pair: AotGraphPair,
    contract: TaskStorageContract | None = None,
) -> SavedValueFootprint:
    """Return semantic storage introduced by one graph-pair alternative."""

    selected = contract or pair.forward.storage_contract
    public_count = pair.forward.output_count - pair.saved_value_count
    if public_count < 0:
        raise ValueError("graph pair has more saved values than forward outputs")

    root_by_id = {root.root_id: root for root in selected.roots}
    public_roots = {
        view.root_id for view in selected.output_views if view.leaf_index < public_count
    }
    saved_roots = {
        view.root_id
        for view in selected.output_views
        if view.leaf_index >= public_count
    }
    boundary = tuple(sorted(saved_roots & public_roots))
    input_roots = tuple(
        sorted(
            root_id
            for root_id in saved_roots - public_roots
            if root_by_id[root_id].kind is StorageRootKind.INPUT
        )
    )
    internal = tuple(
        sorted(
            root_id
            for root_id in saved_roots - public_roots
            if root_by_id[root_id].kind is StorageRootKind.FRESH
        )
    )
    return SavedValueFootprint(
        public_output_count=public_count,
        saved_value_count=pair.saved_value_count,
        input_root_ids=input_roots,
        boundary_root_ids=boundary,
        internal_root_ids=internal,
        input_minimum_bytes=_root_bytes(root_by_id, input_roots),
        boundary_minimum_bytes=_root_bytes(root_by_id, boundary),
        internal_minimum_bytes=_root_bytes(root_by_id, internal),
    )


def _root_bytes(
    roots: dict[int, StorageRoot],
    root_ids: tuple[int, ...],
) -> int:
    return sum(roots[root_id].minimum_span_bytes for root_id in root_ids)


__all__ = ["SavedValueFootprint", "saved_value_footprint"]
