"""What one task's storage contract is made of: roots, views, mutations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from shadowspill.errors import CaptureError
from shadowspill.planner.strict import (
    _integer as strict_integer,
)
from shadowspill.planner.strict import (
    _integer_tuple as strict_integer_tuple,
)
from shadowspill.planner.strict import (
    _optional_integer as strict_optional_integer,
)
from shadowspill.planner.strict import (
    _optional_string as strict_optional_string,
)
from shadowspill.planner.strict import (
    _string as strict_string,
)
from shadowspill.schema import artifact_schema


class StorageRootKind(StrEnum):
    """Semantic origin of one output-reachable task storage."""

    INPUT = "input"
    FRESH = "fresh"


@dataclass(frozen=True, slots=True)
class StorageRoot:
    """One normalized semantic storage root.

    Input roots name the canonical argument position for an input alias
    group. Fresh roots name one FX producer and result index. Physical
    allocation sizes deliberately do not appear in this record.
    """

    root_id: int
    kind: StorageRootKind
    source_input: int | None
    producer_node: str | None
    producer_target: str | None
    producer_result: int | None
    minimum_span_bytes: int

    def __post_init__(self) -> None:
        if self.root_id < 0 or self.minimum_span_bytes < 0:
            raise ValueError("storage-root fields must be non-negative")
        if not isinstance(self.kind, StorageRootKind):
            raise TypeError("storage-root kind has an invalid type")
        if self.kind is StorageRootKind.INPUT:
            if self.source_input is None or self.source_input < 0:
                raise ValueError("input root requires an input position")
            if any(
                value is not None
                for value in (
                    self.producer_node,
                    self.producer_target,
                    self.producer_result,
                )
            ):
                raise ValueError("input root cannot name an FX producer")
        else:
            if self.source_input is not None:
                raise ValueError("fresh root cannot name a task input")
            if not self.producer_node or not self.producer_target:
                raise ValueError("fresh root requires FX producer provenance")
            if self.producer_result is None or self.producer_result < 0:
                raise ValueError("fresh root requires a producer result index")

    def identity(self) -> dict[str, object]:
        return {
            "root_id": self.root_id,
            "kind": self.kind.value,
            "source_input": self.source_input,
            "producer_node": self.producer_node,
            "producer_target": self.producer_target,
            "producer_result": self.producer_result,
            "minimum_span_bytes": self.minimum_span_bytes,
        }


@dataclass(frozen=True, slots=True)
class OutputView:
    """One flattened output view of a semantic storage root."""

    leaf_index: int
    root_id: int
    offset_bytes: int
    span_bytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    layout: str

    def __post_init__(self) -> None:
        if min(self.leaf_index, self.root_id, self.offset_bytes, self.span_bytes) < 0:
            raise ValueError("output-view fields must be non-negative")
        if len(self.shape) != len(self.stride):
            raise ValueError("output-view shape and stride ranks differ")
        if any(extent < 0 for extent in self.shape):
            raise ValueError("output-view shape has a negative extent")
        if any(stride < 0 for stride in self.stride):
            raise ValueError("output-view stride is negative")
        if not self.dtype or not self.layout:
            raise ValueError("output-view dtype and layout must be non-empty")

    def identity(self) -> dict[str, object]:
        return {
            "leaf_index": self.leaf_index,
            "root_id": self.root_id,
            "offset_bytes": self.offset_bytes,
            "span_bytes": self.span_bytes,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "layout": self.layout,
        }


@dataclass(frozen=True, slots=True)
class MutationBinding:
    """One task operation that updates an input storage.

    ``replacement_output_leaf`` distinguishes Export's functional mutation
    form from a dispatcher-schema write.  The executable replacement normally
    has fresh storage and becomes the input object's next authoritative
    generation.  the compiler may prove a no-op update and return the target input
    itself; that preserves the mutation contract without requiring a generation
    replacement.  A schema write has no replacement output because the
    compiled operation writes in place.
    """

    input_position: int
    replacement_output_leaf: int | None
    producer_node: str
    producer_target: str
    argument_name: str

    def __post_init__(self) -> None:
        if self.input_position < 0:
            raise ValueError("mutation input position must be non-negative")
        if (
            self.replacement_output_leaf is not None
            and self.replacement_output_leaf < 0
        ):
            raise ValueError("mutation output leaf must be non-negative")
        if not self.producer_node or not self.producer_target or not self.argument_name:
            raise ValueError("mutation provenance must be non-empty")

    def identity(self) -> dict[str, object]:
        return {
            "input_position": self.input_position,
            "replacement_output_leaf": self.replacement_output_leaf,
            "producer_node": self.producer_node,
            "producer_target": self.producer_target,
            "argument_name": self.argument_name,
        }


@dataclass(frozen=True, slots=True)
class TaskStorageContract:
    """Deterministic semantic storage contract for one functional task contract."""

    roots: tuple[StorageRoot, ...]
    output_views: tuple[OutputView, ...]
    mutations: tuple[MutationBinding, ...]
    compatibility_digest: str

    def __post_init__(self) -> None:
        if tuple(root.root_id for root in self.roots) != tuple(range(len(self.roots))):
            raise ValueError("storage roots must have contiguous indices")
        root_by_id = {root.root_id: root for root in self.roots}
        leaves: set[int] = set()
        used_roots: set[int] = set()
        for view in self.output_views:
            if view.leaf_index in leaves:
                raise ValueError("one output leaf has multiple storage bindings")
            leaves.add(view.leaf_index)
            root = root_by_id.get(view.root_id)
            if root is None:
                raise ValueError("output view references an unknown storage root")
            if view.offset_bytes + view.span_bytes > root.minimum_span_bytes:
                raise ValueError("output view exceeds its semantic storage span")
            used_roots.add(view.root_id)
        if used_roots != set(root_by_id):
            raise ValueError("semantic storage root is not referenced by an output")
        mutation_keys = [
            (
                item.input_position,
                item.producer_node,
                item.argument_name,
            )
            for item in self.mutations
        ]
        if len(set(mutation_keys)) != len(mutation_keys):
            raise ValueError("task storage contract contains duplicate mutations")
        if len(self.compatibility_digest) != 64:
            raise ValueError("task storage contract digest must be SHA-256")

    @staticmethod
    def digest_of(
        roots: tuple[StorageRoot, ...],
        output_views: tuple[OutputView, ...],
        mutations: tuple[MutationBinding, ...],
    ) -> str:
        """What these parts identify, whether or not they make a contract.

        A record read from a store is answered for by its digest before
        anything is built from it, so a record that was changed is rejected
        as changed rather than as whatever its contents then trip over.
        """

        identity = {
            "roots": [root.identity() for root in roots],
            "output_views": [view.identity() for view in output_views],
            "mutations": [mutation.identity() for mutation in mutations],
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @classmethod
    def build(
        cls,
        roots: tuple[StorageRoot, ...],
        output_views: tuple[OutputView, ...],
        mutations: tuple[MutationBinding, ...],
    ) -> TaskStorageContract:
        """One contract over these parts, identified by what they say."""

        return cls(
            roots=roots,
            output_views=output_views,
            mutations=mutations,
            compatibility_digest=cls.digest_of(roots, output_views, mutations),
        )

    def identity(self) -> dict[str, object]:
        return {
            "roots": [root.identity() for root in self.roots],
            "output_views": [view.identity() for view in self.output_views],
            "mutations": [mutation.identity() for mutation in self.mutations],
        }

    def without_device_storage(self, leaves: Collection[int]) -> TaskStorageContract:
        """This contract with the named output leaves holding no storage.

        A contract says what device storage a task's outputs occupy, and it
        is written from a trace. A trace is taken over values that only
        describe the real ones, and a describing value can be wrong about
        where a result lives: an operator whose result is a host scalar is
        traced as device memory of the scalar's size. Nothing on the device
        is ever allocated for it, so no allocation can be found for it
        either, and the contract is what has to give.

        A leaf named here keeps its identity, shape and dtype -- it is still
        one of the task's outputs, still passed from the task that produces
        it to the task that reads it -- and gives up its span. A root whose
        leaves are all named gives up its span with them, which is how the
        rest of the system already says *this needs no storage*: zero bytes,
        no allocation, no residency, no fetch.

        One root is one allocation, so it is in one place. A root named for
        some of its leaves and not others describes an allocation half on
        the device, which no observation can mean, and is refused.
        """

        named = frozenset(leaves)
        unknown = sorted(named - {view.leaf_index for view in self.output_views})
        if unknown:
            raise CaptureError(f"task storage contract has no output leaves {unknown}")
        if not named:
            return self
        emptied = {
            view.root_id for view in self.output_views if view.leaf_index in named
        }
        divided = sorted(
            view.leaf_index
            for view in self.output_views
            if view.root_id in emptied and view.leaf_index not in named
        )
        if divided:
            raise CaptureError(
                "one task storage root is partly off the execution device: "
                f"roots={sorted(emptied)}, leaves_left_on_device={divided}"
            )
        return TaskStorageContract.build(
            tuple(
                replace(root, minimum_span_bytes=0) if root.root_id in emptied else root
                for root in self.roots
            ),
            tuple(
                replace(view, offset_bytes=0, span_bytes=0)
                if view.leaf_index in named
                else view
                for view in self.output_views
            ),
            self.mutations,
        )

    def to_json(self) -> str:
        """Return deterministic standalone diagnostic serialization."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON-compatible contract record."""

        return {
            "schema": artifact_schema("task_storage_contract"),
            "compatibility_digest": self.compatibility_digest,
            **self.identity(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TaskStorageContract:
        """Validate and restore one versioned contract record."""

        expected_keys = {
            "schema",
            "compatibility_digest",
            "roots",
            "output_views",
            "mutations",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "task storage contract fields differ from schema: "
                f"expected={sorted(expected_keys)}, actual={sorted(payload)}"
            )
        if payload["schema"] != artifact_schema("task_storage_contract"):
            raise ValueError("unsupported task storage contract schema")
        roots = tuple(
            _storage_root_from_record(item) for item in _records(payload, "roots")
        )
        output_views = tuple(
            _output_view_from_record(item) for item in _records(payload, "output_views")
        )
        mutations = tuple(
            _mutation_from_record(item) for item in _records(payload, "mutations")
        )
        declared = payload["compatibility_digest"]
        if not isinstance(declared, str) or declared != cls.digest_of(
            roots, output_views, mutations
        ):
            raise ValueError("task storage contract digest does not match its contents")
        return cls.build(roots, output_views, mutations)

    @classmethod
    def from_json(cls, encoded: str) -> TaskStorageContract:
        """Validate and restore deterministic standalone JSON."""

        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("task storage contract is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("task storage contract JSON must contain an object")
        return cls.from_dict(payload)


def _records(
    payload: Mapping[str, object], field: str
) -> tuple[Mapping[str, object], ...]:
    value = payload[field]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"task storage contract {field!r} must be a list of objects")
    return tuple(value)


def _field(field: str) -> str:
    return f"task storage contract field {field!r}"


def _integer(record: Mapping[str, object], field: str) -> int:
    return strict_integer(record.get(field), _field(field))


def _optional_integer(record: Mapping[str, object], field: str) -> int | None:
    return strict_optional_integer(record.get(field), _field(field))


def _optional_string(record: Mapping[str, object], field: str) -> str | None:
    return strict_optional_string(record.get(field), _field(field))


def _string(record: Mapping[str, object], field: str) -> str:
    return strict_string(record.get(field), _field(field))


def _integer_tuple(record: Mapping[str, object], field: str) -> tuple[int, ...]:
    return strict_integer_tuple(record.get(field), _field(field))


def _storage_root_from_record(record: Mapping[str, object]) -> StorageRoot:
    expected = {
        "root_id",
        "kind",
        "source_input",
        "producer_node",
        "producer_target",
        "producer_result",
        "minimum_span_bytes",
    }
    if set(record) != expected:
        raise ValueError("storage-root record fields differ from schema")
    try:
        kind = StorageRootKind(_string(record, "kind"))
    except ValueError as exc:
        raise ValueError("storage-root kind is unknown") from exc
    return StorageRoot(
        _integer(record, "root_id"),
        kind,
        _optional_integer(record, "source_input"),
        _optional_string(record, "producer_node"),
        _optional_string(record, "producer_target"),
        _optional_integer(record, "producer_result"),
        _integer(record, "minimum_span_bytes"),
    )


def _output_view_from_record(record: Mapping[str, object]) -> OutputView:
    expected = {
        "leaf_index",
        "root_id",
        "offset_bytes",
        "span_bytes",
        "shape",
        "stride",
        "dtype",
        "layout",
    }
    if set(record) != expected:
        raise ValueError("output-view record fields differ from schema")
    return OutputView(
        _integer(record, "leaf_index"),
        _integer(record, "root_id"),
        _integer(record, "offset_bytes"),
        _integer(record, "span_bytes"),
        _integer_tuple(record, "shape"),
        _integer_tuple(record, "stride"),
        _string(record, "dtype"),
        _string(record, "layout"),
    )


def _mutation_from_record(record: Mapping[str, object]) -> MutationBinding:
    expected = {
        "input_position",
        "replacement_output_leaf",
        "producer_node",
        "producer_target",
        "argument_name",
    }
    if set(record) != expected:
        raise ValueError("mutation record fields differ from schema")
    return MutationBinding(
        _integer(record, "input_position"),
        _optional_integer(record, "replacement_output_leaf"),
        _string(record, "producer_node"),
        _string(record, "producer_target"),
        _string(record, "argument_name"),
    )
