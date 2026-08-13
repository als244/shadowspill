"""Lower framework-owned task artifacts into canonical ShadowSpill IR."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    MutationSpec,
    ObjectRole,
    ObjectSpec,
    Persistence,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)

from ._live_storage import live_storage_bytes, live_storage_identity
from .capture import GraphArtifact
from .compiled_layout import (
    CompiledTaskLayout,
    reconcile_compiled_task_layout,
    replacement_transition_bytes,
)
from .contracts import CaptureError
from .output_contract import (
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)
from .partition import PartitionedExport, StageExample
from .profiling import TaskMeasurement


@dataclass(frozen=True, slots=True)
class TensorSlot:
    """Position of one tensor leaf in a framework task ABI."""

    leaf_index: int
    object_id: str


@dataclass(frozen=True, slots=True)
class RegistrationBinding:
    """Original registered name associated with one logical tensor view."""

    name: str
    object_id: str
    parameter: bool


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


@dataclass(frozen=True, slots=True)
class TaskEntrypoint:
    """Framework-only executable binding for one canonical task."""

    task_id: str
    module_target: str
    artifact: GraphArtifact
    input_slots: tuple[TensorSlot, ...]
    output_slots: tuple[TensorSlot, ...]
    replacement_output_leaves: tuple[int, ...] = ()
    storage_handoffs: tuple[TaskStorageHandoff, ...] = ()


@dataclass(frozen=True, slots=True)
class LoweredForwardProgram:
    """Canonical forward program plus non-serialized PyTorch bindings."""

    program: Program
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    entrypoints: tuple[TaskEntrypoint, ...]
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[TensorSlot, ...]
    output_tree_spec: TreeSpec
    output_leaf_count: int


@dataclass(frozen=True, slots=True)
class _TensorKey:
    storage_identity: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


@dataclass(slots=True)
class _ObjectRecord:
    object_id: str
    alias_group_id: str
    offset_bytes: int
    size_bytes: int
    role: ObjectRole
    persistence: Persistence


_ROLE_PRIORITY = {
    ObjectRole.OTHER: 0,
    ObjectRole.INPUT: 1,
    ObjectRole.ACTIVATION: 2,
    ObjectRole.OUTPUT: 3,
    ObjectRole.GRADIENT: 3,
    ObjectRole.OPTIMIZER_STATE: 4,
    ObjectRole.BUFFER: 5,
    ObjectRole.PARAMETER: 6,
}


def _view_extent_bytes(tensor: torch.Tensor) -> int:
    """Return the compact storage span needed for one strided tensor view."""

    if tensor.numel() == 0:
        return 0
    if any(stride < 0 for stride in tensor.stride()):
        raise CaptureError("compiled task output has a negative stride")
    last_element = sum(
        (extent - 1) * stride
        for extent, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return int((last_element + 1) * tensor.element_size())


def _contract_view_size(view: OutputView) -> int:
    """Return logical bytes from one already-validated symbolic output view."""

    elements = math.prod(view.shape)
    if elements == 0:
        return 0
    span_elements = 1 + sum(
        (extent - 1) * stride
        for extent, stride in zip(view.shape, view.stride, strict=True)
    )
    if span_elements <= 0 or view.span_bytes % span_elements:
        raise CaptureError("task output view has an invalid symbolic byte span")
    return elements * (view.span_bytes // span_elements)


class ObjectCatalog:
    """Canonical cross-task objects and alias bundles.

    Live framework storage identities are accepted only by ``add`` for
    user-owned inputs, parameters, buffers, and their views. Task outputs enter
    through ``TaskBindingResolver`` using the offline semantic contract and
    reconciled compiled layout.
    """

    def __init__(self, *, device_id: str) -> None:
        self._device_id = device_id
        self._tensor_keepalive: list[torch.Tensor] = []
        self._alias_by_storage: dict[int, str] = {}
        self._alias_sizes: dict[str, int] = {}
        self._retain_host: set[str] = set()
        self._object_by_key: dict[_TensorKey, str] = {}
        self._objects: list[_ObjectRecord] = []
        self._object_specs: tuple[ObjectSpec, ...] | None = None
        self._next_alias_id = 0

    @staticmethod
    def key(tensor: torch.Tensor) -> _TensorKey:
        if tensor.layout is not torch.strided:
            raise CaptureError("program lowering currently requires strided tensors")
        return _TensorKey(
            storage_identity=live_storage_identity(tensor),
            storage_offset=int(tensor.storage_offset()),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
        )

    def add(
        self,
        tensor: torch.Tensor,
        *,
        role: ObjectRole,
        persistence: Persistence,
        retain_spill_copy: bool = False,
    ) -> str:
        return self._add(
            tensor,
            role=role,
            persistence=persistence,
            retain_spill_copy=retain_spill_copy,
        )

    def add_output_view(
        self,
        tensor: torch.Tensor,
        *,
        alias_id: str,
        offset_bytes: int,
        role: ObjectRole,
        persistence: Persistence,
    ) -> str:
        """Create one graph-declared view in an existing residency bundle."""

        if alias_id not in self._alias_sizes:
            raise CaptureError("task output references an unknown alias group")
        if offset_bytes + _view_extent_bytes(tensor) > self._alias_sizes[alias_id]:
            raise CaptureError("task output view exceeds its declared storage")
        self._tensor_keepalive.append(tensor)
        return self._new_object(
            self.key(tensor),
            alias_id=alias_id,
            offset_bytes=offset_bytes,
            tensor=tensor,
            role=role,
            persistence=persistence,
            index_by_tensor_key=False,
        )

    def add_contract_output_view(
        self,
        view: OutputView,
        *,
        alias_id: str,
        offset_bytes: int,
        role: ObjectRole,
        persistence: Persistence,
    ) -> str:
        """Create an output object directly from the offline task contract."""

        if alias_id not in self._alias_sizes:
            raise CaptureError("task output references an unknown alias group")
        if offset_bytes + view.span_bytes > self._alias_sizes[alias_id]:
            raise CaptureError("task output view exceeds its declared storage")
        return self._new_record(
            alias_id=alias_id,
            offset_bytes=offset_bytes,
            size_bytes=_contract_view_size(view),
            role=role,
            persistence=persistence,
        )

    def validate_canonical_output_view(
        self,
        tensor: torch.Tensor,
        object_id: str,
        *,
        alias_id: str,
        offset_bytes: int,
    ) -> None:
        """Validate a compiled output against an existing canonical object.

        This deliberately never merges alias groups.  Compiled input reuse is
        a task-local lease handoff, not proof that two cross-task objects share
        one permanent residency bundle.
        """

        self._tensor_keepalive.append(tensor)
        record = self._record(object_id)
        if record.alias_group_id != alias_id:
            raise CaptureError("canonical output changed its alias group")
        if offset_bytes + _view_extent_bytes(tensor) > self._alias_sizes[alias_id]:
            raise CaptureError("canonical output exceeds its declared storage")
        if record.offset_bytes != offset_bytes:
            raise CaptureError(
                "canonical output changed its storage offset: "
                f"object={object_id}, expected={record.offset_bytes}, "
                f"actual={offset_bytes}"
            )
        size_bytes = int(tensor.numel()) * tensor.element_size()
        if record.size_bytes != size_bytes:
            raise CaptureError(
                "canonical output changed its logical byte size: "
                f"object={object_id}, expected={record.size_bytes}, "
                f"actual={size_bytes}"
            )

    def validate_canonical_contract_view(
        self,
        view: OutputView,
        object_id: str,
        *,
        alias_id: str,
        offset_bytes: int,
    ) -> None:
        """Validate a contract view against an existing canonical object."""

        record = self._record(object_id)
        if record.alias_group_id != alias_id:
            raise CaptureError("canonical output changed its alias group")
        if offset_bytes + view.span_bytes > self._alias_sizes[alias_id]:
            raise CaptureError("canonical output exceeds its declared storage")
        if record.offset_bytes != offset_bytes:
            raise CaptureError(
                "canonical output changed its storage offset: "
                f"object={object_id}, expected={record.offset_bytes}, "
                f"actual={offset_bytes}"
            )
        size_bytes = _contract_view_size(view)
        if record.size_bytes != size_bytes:
            raise CaptureError(
                "canonical output changed its logical byte size: "
                f"object={object_id}, expected={record.size_bytes}, "
                f"actual={size_bytes}"
            )

    def _add(
        self,
        tensor: torch.Tensor,
        *,
        role: ObjectRole,
        persistence: Persistence,
        retain_spill_copy: bool,
    ) -> str:
        self._tensor_keepalive.append(tensor)
        key = self.key(tensor)
        object_id = self._object_by_key.get(key)
        if object_id is not None:
            self._upgrade_object(
                object_id,
                role=role,
                persistence=persistence,
                retain_spill_copy=retain_spill_copy,
            )
            return object_id
        alias_id = self._alias_by_storage.get(key.storage_identity)
        storage_bytes = live_storage_bytes(tensor)
        if alias_id is None:
            alias_id = self._new_alias(storage_bytes)
            self._alias_by_storage[key.storage_identity] = alias_id
        elif self._alias_sizes[alias_id] != storage_bytes:
            raise CaptureError("one capture storage reported inconsistent byte extents")
        if retain_spill_copy:
            self._retain_host.add(alias_id)
        return self._new_object(
            key,
            alias_id=alias_id,
            offset_bytes=int(tensor.storage_offset()) * tensor.element_size(),
            tensor=tensor,
            role=role,
            persistence=persistence,
        )

    def _new_alias(self, size_bytes: int) -> str:
        alias_id = f"alias_{self._next_alias_id:06d}"
        self._next_alias_id += 1
        self._alias_sizes[alias_id] = size_bytes
        return alias_id

    def new_output_alias(self, size_bytes: int) -> str:
        """Create one task-local storage bundle declared by the FX graph."""

        return self._new_alias(size_bytes)

    def alias_size(self, alias_id: str) -> int:
        """Return one residency bundle's declared physical extent."""

        try:
            return self._alias_sizes[alias_id]
        except KeyError as exc:
            raise CaptureError("task references an unknown alias group") from exc

    def _new_object(
        self,
        key: _TensorKey,
        *,
        alias_id: str,
        offset_bytes: int,
        tensor: torch.Tensor,
        role: ObjectRole,
        persistence: Persistence,
        index_by_tensor_key: bool = True,
    ) -> str:
        object_id = self._new_record(
            alias_id=alias_id,
            offset_bytes=offset_bytes,
            size_bytes=int(tensor.numel()) * tensor.element_size(),
            role=role,
            persistence=persistence,
        )
        if index_by_tensor_key:
            self._object_by_key[key] = object_id
        return object_id

    def _new_record(
        self,
        *,
        alias_id: str,
        offset_bytes: int,
        size_bytes: int,
        role: ObjectRole,
        persistence: Persistence,
    ) -> str:
        object_id = f"object_{len(self._objects):06d}"
        self._objects.append(
            _ObjectRecord(
                object_id=object_id,
                alias_group_id=alias_id,
                offset_bytes=offset_bytes,
                size_bytes=size_bytes,
                role=role,
                persistence=persistence,
            )
        )
        self._object_specs = None
        return object_id

    def _upgrade_object(
        self,
        object_id: str,
        *,
        role: ObjectRole,
        persistence: Persistence,
        retain_spill_copy: bool,
    ) -> None:
        record = self._record(object_id)
        changed = False
        if (
            persistence is Persistence.CHECKPOINT
            and record.persistence is not Persistence.CHECKPOINT
        ):
            record.persistence = persistence
            changed = True
        if _ROLE_PRIORITY[role] > _ROLE_PRIORITY[record.role]:
            record.role = role
            changed = True
        if retain_spill_copy:
            self._retain_host.add(record.alias_group_id)
        if changed:
            self._object_specs = None

    def alias_id(self, object_id: str) -> str:
        return self._record(object_id).alias_group_id

    def object_size(self, object_id: str) -> int:
        return self._record(object_id).size_bytes

    def mark_output(self, object_id: str) -> None:
        record = self._record(object_id)
        if record.role in {ObjectRole.OTHER, ObjectRole.ACTIVATION}:
            record.role = ObjectRole.OUTPUT
            self._object_specs = None

    def _record(self, object_id: str) -> _ObjectRecord:
        index = int(object_id.removeprefix("object_"))
        return self._objects[index]

    def alias_groups(self) -> tuple[AliasGroupSpec, ...]:
        return tuple(
            AliasGroupSpec(
                alias_group_id,
                self._device_id,
                size_bytes,
                retain_spill_copy=alias_group_id in self._retain_host,
            )
            for alias_group_id, size_bytes in self._alias_sizes.items()
        )

    def objects(self) -> tuple[ObjectSpec, ...]:
        if self._object_specs is None:
            self._object_specs = tuple(
                ObjectSpec(
                    item.object_id,
                    item.alias_group_id,
                    item.offset_bytes,
                    item.size_bytes,
                    item.role,
                    item.persistence,
                )
                for item in self._objects
            )
        return self._object_specs


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
        replacement_object = self._replacement_by_leaf.get(leaf_index)
        source_object = (
            self._resolve_input_object(root.source_input)
            if root.kind is StorageRootKind.INPUT and root.source_input is not None
            else None
        )
        if replacement_object is not None:
            if (
                canonical_object_id is not None
                and canonical_object_id != replacement_object
            ):
                raise CaptureError(
                    "functional mutation output conflicts with canonical binding"
                )
            canonical_object_id = replacement_object
            alias_id = self._inventory.alias_id(replacement_object)
            compiled_root = self._layout.root(root.root_id)
            if compiled_root.requested_bytes < self._inventory.alias_size(alias_id):
                raise CaptureError(
                    "functional mutation allocation is smaller than its target storage"
                )
            self._alias_by_fresh_root[root.root_id] = alias_id
        elif canonical_object_id is not None:
            alias_id = self._inventory.alias_id(canonical_object_id)
            if source_object is not None:
                source_alias = self._inventory.alias_id(source_object)
                if source_alias != alias_id:
                    source_bytes = self._inventory.alias_size(source_alias)
                    destination_bytes = self._inventory.alias_size(alias_id)
                    if source_bytes != destination_bytes:
                        raise CaptureError(
                            "task-local storage handoff changes physical extent: "
                            f"source={source_object} ({source_bytes}), "
                            f"destination={canonical_object_id} "
                            f"({destination_bytes})"
                        )
                    existing_handoff = self._handoff_by_destination.get(alias_id)
                    handoff = TaskStorageHandoff(
                        leaf_index, source_object, canonical_object_id
                    )
                    if (
                        existing_handoff is not None
                        and existing_handoff.source_object_id != source_object
                    ):
                        raise CaptureError(
                            "one task output receives storage from multiple inputs"
                        )
                    self._handoff_by_destination.setdefault(alias_id, handoff)
            elif root.kind is StorageRootKind.FRESH:
                compiled_root = self._layout.root(root.root_id)
                if compiled_root.charged_bytes != self._inventory.alias_size(alias_id):
                    raise CaptureError(
                        "compiled output allocation differs from its canonical "
                        "physical extent"
                    )
                self._alias_by_fresh_root[root.root_id] = alias_id
        else:
            alias_id = self._resolve_alias(root)
        compiled_view = next(
            item for item in self._layout.output_views if item.leaf_index == leaf_index
        )
        offset_bytes = (
            view.offset_bytes
            if root.kind is StorageRootKind.INPUT
            else compiled_view.offset_bytes
        )
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
        self._view_by_identity[view_identity] = object_id
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
        abi_position = self._artifact_input_position[contract_position]
        try:
            return self._input_by_position[abi_position]
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
    if len(stage.input_sources) != len(input_leaves):
        raise CaptureError("stage input provenance arity changed")
    slots: list[TensorSlot] = []
    for compact_index, stage_position in enumerate(artifact.tensor_argument_positions):
        if stage_position >= len(input_leaves) or not isinstance(
            input_leaves[stage_position], torch.Tensor
        ):
            raise CaptureError("stage tensor argument position is invalid")
        source = stage.input_sources[stage_position]
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


def lower_forward_program(
    model: nn.Module,
    partitioned: PartitionedExport,
    artifacts: tuple[GraphArtifact, ...],
    measurements: tuple[TaskMeasurement, ...],
    *,
    storage_contracts: Mapping[str, TaskStorageContract] | None = None,
    device_ordinal: int = 0,
) -> LoweredForwardProgram:
    """Create one deterministic canonical program from forward task positions."""

    stage_count = len(partitioned.stages)
    if len(artifacts) != stage_count or len(measurements) != stage_count:
        raise CaptureError("stage, artifact, and measurement counts must match")
    device_id = f"cuda_{device_ordinal}"
    inventory = ObjectCatalog(device_id=device_id)
    registrations: list[RegistrationBinding] = []
    checkpoint_names = set(model.state_dict())
    for name, parameter in model.named_parameters(remove_duplicate=False):
        object_id = inventory.add(
            parameter,
            role=ObjectRole.PARAMETER,
            persistence=Persistence.CHECKPOINT,
            retain_spill_copy=True,
        )
        registrations.append(RegistrationBinding(name, object_id, True))
    parameter_names = {
        name for name, _ in model.named_parameters(remove_duplicate=False)
    }
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in parameter_names:
            continue
        object_id = inventory.add(
            buffer,
            role=ObjectRole.BUFFER,
            persistence=(
                Persistence.CHECKPOINT if name in checkpoint_names else Persistence.RUN
            ),
            retain_spill_copy=True,
        )
        registrations.append(RegistrationBinding(name, object_id, False))

    root_input_slots: list[TensorSlot] = []
    for position, value in enumerate(partitioned.root_inputs):
        if not isinstance(value, torch.Tensor):
            continue
        object_id = inventory.add(
            value,
            role=ObjectRole.INPUT,
            persistence=Persistence.STEP,
            retain_spill_copy=True,
        )
        root_input_slots.append(TensorSlot(position, object_id))
    root_objects = {slot.leaf_index: slot.object_id for slot in root_input_slots}

    profiles: list[TaskProfile] = []
    profile_by_key: dict[tuple[object, ...], str] = {}
    profile_ids: list[str] = []
    contracts = tuple(
        (
            artifact.storage_contract
            if storage_contracts is None
            else storage_contracts[artifact.compatibility_digest]
        )
        for artifact in artifacts
    )
    compiled_layouts = tuple(
        reconcile_compiled_task_layout(contract, measurement)
        for contract, measurement in zip(contracts, measurements, strict=True)
    )
    for artifact, contract, measurement, layout in zip(
        artifacts, contracts, measurements, compiled_layouts, strict=True
    ):
        transition_bytes = replacement_transition_bytes(contract, layout)
        key = (
            artifact.compatibility_digest,
            measurement.runtime_ns,
            measurement.workspace_charged_bytes,
            transition_bytes,
        )
        profile_id = profile_by_key.get(key)
        if profile_id is None:
            profile_id = f"profile_{len(profiles):06d}"
            profile_by_key[key] = profile_id
            profiles.append(
                TaskProfile(
                    profile_id,
                    measurement.runtime_ns,
                    measurement.workspace_charged_bytes + transition_bytes,
                    artifact.compatibility_digest,
                )
            )
        profile_ids.append(profile_id)

    tasks: list[TaskSpec] = []
    entrypoints: list[TaskEntrypoint] = []
    produced_aliases: set[str] = set()
    last_output_objects: list[str] = []
    stage_output_objects: list[dict[int, str]] = []
    for index, (stage, artifact, contract, profile_id) in enumerate(
        zip(partitioned.stages, artifacts, contracts, profile_ids, strict=True)
    ):
        input_slots = list(
            resolve_stage_input_slots(
                stage,
                artifact,
                root_objects=root_objects,
                stage_outputs=tuple(stage_output_objects),
                compact_leaf_indices=False,
            )
        )
        input_objects = list(dict.fromkeys(slot.object_id for slot in input_slots))

        input_aliases = {inventory.alias_id(value) for value in input_objects}
        output_leaves, _ = tree_flatten(stage.output)
        layout = compiled_layouts[index]
        output_resolver = TaskBindingResolver(
            inventory,
            artifact,
            tuple(input_slots),
            layout,
            storage_contract=contract,
        )
        output_slots: list[TensorSlot] = []
        output_objects: list[str] = []
        for position, leaf in enumerate(output_leaves):
            if not isinstance(leaf, torch.Tensor):
                continue
            object_id = output_resolver.bind(
                position,
                leaf,
                role=ObjectRole.ACTIVATION,
                persistence=Persistence.STEP,
            )
            output_slots.append(TensorSlot(position, object_id))
            output_alias = inventory.alias_id(object_id)
            if (
                object_id not in input_objects
                and output_alias not in input_aliases
                and object_id not in output_objects
            ):
                output_objects.append(object_id)
                produced_aliases.add(output_alias)
            if position in stage.user_output_indices:
                inventory.mark_output(object_id)
                last_output_objects.append(object_id)
        stage_output_objects.append(
            {slot.leaf_index: slot.object_id for slot in output_slots}
        )
        task_id = f"task_{index:06d}"
        dependencies = () if index == 0 else (f"task_{index - 1:06d}",)
        tasks.append(
            TaskSpec(
                task_id,
                ResourceSpec(device_id, ResourceKind.COMPUTE),
                profile_id,
                dependencies=dependencies,
                inputs=tuple(input_objects),
                outputs=tuple(output_objects),
                mutations=tuple(
                    MutationSpec(object_id)
                    for object_id in output_resolver.mutation_object_ids
                ),
                phase="forward",
            )
        )
        entrypoints.append(
            TaskEntrypoint(
                task_id,
                stage.module_target,
                artifact,
                tuple(input_slots),
                tuple(output_slots),
                output_resolver.replacement_output_leaves,
                output_resolver.storage_handoffs,
            )
        )

    alias_groups = inventory.alias_groups()
    objects = inventory.objects()
    initial_aliases = {
        item.alias_group_id
        for item in objects
        if item.role in {ObjectRole.PARAMETER, ObjectRole.BUFFER, ObjectRole.INPUT}
        and item.alias_group_id not in produced_aliases
    }
    final_host = {
        item.alias_group_id
        for item in objects
        if item.persistence is Persistence.CHECKPOINT or item.role is ObjectRole.INPUT
    }
    final_device = {inventory.alias_id(object_id) for object_id in last_output_objects}
    final_host -= final_device
    initial_residency = tuple(
        ResidencySpec(group.alias_group_id, MemoryLocation.HOST)
        for group in alias_groups
        if group.alias_group_id in initial_aliases
    )
    final_residency = tuple(
        ResidencySpec(
            group.alias_group_id,
            MemoryLocation.DEVICE
            if group.alias_group_id in final_device
            else MemoryLocation.HOST,
        )
        for group in alias_groups
        if group.alias_group_id in final_host | final_device
    )
    program = Program(
        devices=(DeviceSpec(device_id, "process_0", "cuda", device_ordinal),),
        alias_groups=alias_groups,
        objects=objects,
        profiles=tuple(profiles),
        tasks=tuple(tasks),
    )
    _last_leaves, last_tree_spec = tree_flatten(partitioned.stages[-1].output)
    return LoweredForwardProgram(
        program,
        initial_residency,
        final_residency,
        tuple(entrypoints),
        tuple(registrations),
        tuple(root_input_slots),
        last_tree_spec,
        len(_last_leaves),
    )


__all__ = [
    "LoweredForwardProgram",
    "RegistrationBinding",
    "TaskEntrypoint",
    "TaskStorageHandoff",
    "TensorSlot",
    "lower_forward_program",
]
