"""Resolve one task's semantic storage contract into canonical objects."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.ir import ObjectRole, Persistence
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import (
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)
from shadowspill.pytorch.compilation.layout import CompiledTaskLayout

from ..partition import StageExample
from .catalog import ObjectCatalog, TensorSlot, _view_extent_bytes


@dataclass(frozen=True, slots=True)
class TaskStorageHandoff:
    """Transfer one task-input lease to a distinct returned logical object.

    Inductor may return an input allocation for a logically distinct output.
    The relationship is local to this invocation: it must not merge the two
    objects' alias groups globally.  A handoff is legal only when the selected
    schedule releases ``source_object_id`` at the same task boundary.
    """

    leaf_index: int
    source_object_id: str
    destination_object_id: str


class TaskBindingResolver:
    """Bind one task contract and compiled layout into canonical objects."""

    def __init__(
        self,
        inventory: ObjectCatalog,
        artifact: GraphArtifact,
        input_slots: tuple[TensorSlot, ...],
        layout: CompiledTaskLayout,
        *,
        storage_contract: TaskStorageContract | None = None,
    ) -> None:
        self._inventory = inventory
        contract = storage_contract or artifact.storage_contract
        if layout.contract_digest != contract.compatibility_digest:
            raise CaptureError("compiled task layout belongs to another contract")
        self._layout = layout
        self._views = {item.leaf_index: item for item in contract.output_views}
        self._roots = {item.root_id: item for item in contract.roots}
        self._input_by_position = {
            item.leaf_index: item.object_id for item in input_slots
        }
        self._artifact_input_position = artifact.tensor_argument_positions
        self._alias_by_fresh_root: dict[int, str] = {}
        self._replacement_by_leaf = {
            item.replacement_output_leaf: self._resolve_input_object(
                item.input_position
            )
            for item in contract.mutations
            if item.replacement_output_leaf is not None
            and self._root_for_leaf(item.replacement_output_leaf).kind
            is StorageRootKind.FRESH
        }
        self._mutation_objects = tuple(
            dict.fromkeys(
                self._resolve_input_object(item.input_position)
                for item in contract.mutations
            )
        )
        self._view_by_identity: dict[
            tuple[str, int, tuple[int, ...], tuple[int, ...], str], str
        ] = {}
        self._handoff_by_destination: dict[str, TaskStorageHandoff] = {}

    def bind(
        self,
        leaf_index: int,
        tensor: torch.Tensor,
        *,
        role: ObjectRole,
        persistence: Persistence,
        canonical_object_id: str | None = None,
    ) -> str:
        """Bind one returned tensor and validate it against the contract."""

        view = self._views.get(leaf_index)
        if view is None:
            raise CaptureError(
                f"tensor output leaf {leaf_index} has no graph storage contract"
            )
        if (
            tuple(tensor.shape) != view.shape
            or tuple(tensor.stride()) != view.stride
            or str(tensor.dtype) != view.dtype
            or _view_extent_bytes(tensor) != view.span_bytes
        ):
            raise CaptureError("task output tensor differs from its storage contract")
        return self._bind_view(
            leaf_index,
            view,
            tensor=tensor,
            role=role,
            persistence=persistence,
            canonical_object_id=canonical_object_id,
        )

    def bind_contract(
        self,
        leaf_index: int,
        *,
        role: ObjectRole,
        persistence: Persistence,
        canonical_object_id: str | None = None,
    ) -> str:
        """Bind one output using only its offline semantic/physical contract."""

        view = self._views.get(leaf_index)
        if view is None:
            raise CaptureError(
                f"tensor output leaf {leaf_index} has no graph storage contract"
            )
        return self._bind_view(
            leaf_index,
            view,
            tensor=None,
            role=role,
            persistence=persistence,
            canonical_object_id=canonical_object_id,
        )

    def _bind_view(
        self,
        leaf_index: int,
        view: OutputView,
        *,
        tensor: torch.Tensor | None,
        role: ObjectRole,
        persistence: Persistence,
        canonical_object_id: str | None,
    ) -> str:
        root = self._roots[view.root_id]
        alias_id, canonical_object_id = self._bind_output_alias(
            leaf_index,
            root,
            canonical_object_id,
        )
        compiled_view = next(
            item for item in self._layout.output_views if item.leaf_index == leaf_index
        )
        offset_bytes = (
            view.offset_bytes
            if root.kind is StorageRootKind.INPUT
            else compiled_view.offset_bytes
        )
        object_id = self._bind_output_object(
            view,
            tensor,
            alias_id=alias_id,
            offset_bytes=offset_bytes,
            role=role,
            persistence=persistence,
            canonical_object_id=canonical_object_id,
        )
        self._view_by_identity[
            (alias_id, offset_bytes, view.shape, view.stride, view.dtype)
        ] = object_id
        return object_id

    def _bind_output_alias(
        self,
        leaf_index: int,
        root: StorageRoot,
        canonical_object_id: str | None,
    ) -> tuple[str, str | None]:
        replacement_object = self._replacement_by_leaf.get(leaf_index)
        if replacement_object is not None:
            if (
                canonical_object_id is not None
                and canonical_object_id != replacement_object
            ):
                raise CaptureError(
                    "functional mutation output conflicts with canonical binding"
                )
            canonical_object_id = replacement_object
        if canonical_object_id is None:
            return self._resolve_alias(root), None
        alias_id = self._inventory.alias_id(canonical_object_id)
        if replacement_object is not None:
            compiled_root = self._layout.root(root.root_id)
            if compiled_root.requested_bytes < self._inventory.alias_size(alias_id):
                raise CaptureError(
                    "functional mutation allocation is smaller than its target storage"
                )
            self._alias_by_fresh_root[root.root_id] = alias_id
            return alias_id, canonical_object_id
        self._validate_canonical_alias(
            leaf_index,
            root,
            canonical_object_id,
            alias_id,
        )
        return alias_id, canonical_object_id

    def _validate_canonical_alias(
        self,
        leaf_index: int,
        root: StorageRoot,
        canonical_object_id: str,
        alias_id: str,
    ) -> None:
        source_object = (
            self._resolve_input_object(root.source_input)
            if root.kind is StorageRootKind.INPUT and root.source_input is not None
            else None
        )
        if source_object is not None:
            self._register_storage_handoff(
                leaf_index,
                source_object,
                canonical_object_id,
                alias_id,
            )
            return
        if root.kind is StorageRootKind.FRESH:
            compiled_root = self._layout.root(root.root_id)
            if compiled_root.charged_bytes != self._inventory.alias_size(alias_id):
                raise CaptureError(
                    "compiled output allocation differs from its canonical "
                    "physical extent"
                )
            self._alias_by_fresh_root[root.root_id] = alias_id

    def _register_storage_handoff(
        self,
        leaf_index: int,
        source_object: str,
        destination_object: str,
        destination_alias: str,
    ) -> None:
        source_alias = self._inventory.alias_id(source_object)
        if source_alias == destination_alias:
            return
        source_bytes = self._inventory.alias_size(source_alias)
        destination_bytes = self._inventory.alias_size(destination_alias)
        if source_bytes != destination_bytes:
            raise CaptureError(
                "task-local storage handoff changes physical extent: "
                f"source={source_object} ({source_bytes}), "
                f"destination={destination_object} ({destination_bytes})"
            )
        existing = self._handoff_by_destination.get(destination_alias)
        handoff = TaskStorageHandoff(
            leaf_index,
            source_object,
            destination_object,
        )
        if existing is not None and existing.source_object_id != source_object:
            raise CaptureError("one task output receives storage from multiple inputs")
        self._handoff_by_destination.setdefault(destination_alias, handoff)

    def _bind_output_object(
        self,
        view: OutputView,
        tensor: torch.Tensor | None,
        *,
        alias_id: str,
        offset_bytes: int,
        role: ObjectRole,
        persistence: Persistence,
        canonical_object_id: str | None,
    ) -> str:
        view_identity = (
            alias_id,
            offset_bytes,
            view.shape,
            view.stride,
            view.dtype,
        )
        existing = self._view_by_identity.get(view_identity)
        if canonical_object_id is not None:
            object_id = canonical_object_id
            if existing is not None and existing != object_id:
                raise CaptureError(
                    "one compiled output view maps to multiple canonical objects"
                )
            if tensor is None:
                self._inventory.validate_canonical_contract_view(
                    view,
                    object_id,
                    alias_id=alias_id,
                    offset_bytes=offset_bytes,
                )
            else:
                self._inventory.validate_canonical_output_view(
                    tensor,
                    object_id,
                    alias_id=alias_id,
                    offset_bytes=offset_bytes,
                )
        elif existing is not None:
            object_id = existing
        else:
            object_id = (
                self._inventory.add_contract_output_view(
                    view,
                    alias_id=alias_id,
                    offset_bytes=offset_bytes,
                    role=role,
                    persistence=persistence,
                )
                if tensor is None
                else self._inventory.add_output_view(
                    tensor,
                    alias_id=alias_id,
                    offset_bytes=offset_bytes,
                    role=role,
                    persistence=persistence,
                )
            )
        return object_id

    @property
    def mutation_object_ids(self) -> tuple[str, ...]:
        """Canonical objects written or replaced by this task."""

        return self._mutation_objects

    @property
    def replacement_output_leaves(self) -> tuple[int, ...]:
        """Output leaves whose fresh allocation replaces input state."""

        return tuple(sorted(self._replacement_by_leaf))

    @property
    def storage_handoffs(self) -> tuple[TaskStorageHandoff, ...]:
        """Task-local input-to-output lease transfers, ordered by leaf."""

        return tuple(
            sorted(
                self._handoff_by_destination.values(),
                key=lambda item: item.leaf_index,
            )
        )

    def _resolve_input_object(self, contract_position: int) -> str:
        if contract_position >= len(self._artifact_input_position):
            raise CaptureError("task mutation has an invalid input position")
        artifact_position = self._artifact_input_position[contract_position]
        try:
            return self._input_by_position[artifact_position]
        except KeyError as exc:
            raise CaptureError(
                "task mutation target is absent from its tensor input slots"
            ) from exc

    def _root_for_leaf(self, leaf_index: int) -> StorageRoot:
        view = self._views.get(leaf_index)
        if view is None:
            raise CaptureError(
                f"functional mutation output leaf {leaf_index} has no storage root"
            )
        return self._roots[view.root_id]

    def _resolve_alias(self, root: StorageRoot) -> str:
        if root.kind is StorageRootKind.FRESH:
            existing = self._alias_by_fresh_root.get(root.root_id)
            if existing is not None:
                return existing
            compiled_root = self._layout.root(root.root_id)
            alias_id = self._inventory.new_output_alias(compiled_root.charged_bytes)
            self._alias_by_fresh_root[root.root_id] = alias_id
            return alias_id
        if root.source_input is None:
            raise CaptureError("task output has an invalid input-root position")
        source = self._resolve_input_object(root.source_input)
        alias_id = self._inventory.alias_id(source)
        if root.minimum_span_bytes > self._inventory.alias_size(alias_id):
            raise CaptureError(
                "task input-root output exceeds its canonical alias storage"
            )
        return alias_id


def resolve_stage_input_slots(
    stage: StageExample,
    artifact: GraphArtifact,
    *,
    root_objects: dict[int, str],
    stage_outputs: tuple[dict[int, str], ...],
    compact_leaf_indices: bool,
) -> tuple[TensorSlot, ...]:
    """Resolve stage inputs from split-root FX topology, never storage IDs."""

    input_leaves, _ = tree_flatten(stage.inputs)
    if len(stage.stage.input_sources) != len(input_leaves):
        raise CaptureError("stage input provenance arity changed")
    slots: list[TensorSlot] = []
    for compact_index, stage_position in enumerate(artifact.tensor_argument_positions):
        if stage_position >= len(input_leaves) or not isinstance(
            input_leaves[stage_position], torch.Tensor
        ):
            raise CaptureError("stage tensor argument position is invalid")
        source = stage.stage.input_sources[stage_position]
        if source is None:
            raise CaptureError("tensor stage argument has no semantic source")
        if source.root_input_index is not None:
            object_id = root_objects.get(source.root_input_index)
            if object_id is None:
                raise CaptureError("stage references an unregistered root input")
        else:
            assert source.producer_stage_index is not None
            assert source.producer_output_index is not None
            try:
                object_id = stage_outputs[source.producer_stage_index][
                    source.producer_output_index
                ]
            except (IndexError, KeyError) as exc:
                raise CaptureError(
                    "stage references an unavailable producer output"
                ) from exc
        slots.append(
            TensorSlot(
                compact_index if compact_leaf_indices else stage_position,
                object_id,
            )
        )
    return tuple(slots)


__all__ = ["TaskBindingResolver", "TaskStorageHandoff", "resolve_stage_input_slots"]
