"""What one task's storage contract is made of: roots, views, mutations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from shadowspill.errors import CaptureError
from shadowspill.planning.memory_search.strict import _integer as strict_integer
from shadowspill.planning.memory_search.strict import (
    _integer_tuple as strict_integer_tuple,
)
from shadowspill.planning.memory_search.strict import (
    _optional_integer as strict_optional_integer,
)
from shadowspill.planning.memory_search.strict import (
    _optional_string as strict_optional_string,
)
from shadowspill.planning.memory_search.strict import _string as strict_string
from shadowspill.schema import artifact_schema


class StorageRootKind(StrEnum):
    """Where one storage a task's outputs reach comes from.

    Attributes
    ----------
    INPUT : str
        The storage of one of the task's inputs.
    FRESH : str
        Storage the task allocates.
    """

    INPUT = "input"
    FRESH = "fresh"


@dataclass(frozen=True, slots=True)
class StorageRoot:
    """One storage a task's outputs reach.

    A root names where the storage comes from, never how large an allocation
    is: an input root names its input, a fresh root the graph node that
    produces it.

    Parameters
    ----------
    root_id : int
        The root's position in the contract's roots.
    kind : StorageRootKind
        Where the storage comes from.
    source_input : int or None
        The input's position, for an input root; None for a fresh root.
    producer_node : str or None
        Name of the graph node that produces a fresh root; None for an input
        root.
    producer_target : str or None
        The operation that node calls, as text; None for an input root.
    producer_result : int or None
        Which of the node's results the root is; None for an input root.
    minimum_span_bytes : int
        Bytes from the root's start that its output views reach.

    Raises
    ------
    ValueError
        If ``root_id`` or ``minimum_span_bytes`` is negative, or the fields
        given do not match ``kind``.
    TypeError
        If ``kind`` is not a `StorageRootKind`.
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
        """Return the record as plain values, as its contract's digest covers it.

        Returns
        -------
        dict
            Its fields by name.
        """
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
    """One output of a task, as a view of one storage root.

    Parameters
    ----------
    leaf_index : int
        The output's position among the task's flattened outputs.
    root_id : int
        The root it views.
    offset_bytes : int
        Where the view starts in the root, in bytes.
    span_bytes : int
        Bytes of the root the view reaches from ``offset_bytes``, which can
        differ from its element count times element size.
    shape : tuple of int
        The extent of each dimension.
    stride : tuple of int
        Elements between neighbours along each dimension.
    dtype : str
        The element type, as the frontend names it.
    layout : str
        The memory layout, as the frontend names it.

    Raises
    ------
    ValueError
        If a position, offset, span, extent or stride is negative, ``shape``
        and ``stride`` differ in length, or ``dtype`` or ``layout`` is empty.
    """

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
        """Return the record as plain values, as its contract's digest covers it.

        Returns
        -------
        dict
            Its fields by name.
        """
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
    """One operation of a task that updates an input's storage.

    An update either returns a replacement output, which becomes the input
    object's next generation, or writes the input in place and returns nothing
    for it. A replacement normally has storage of its own; when the compiler
    finds the update has no effect, it may return the input itself.

    Parameters
    ----------
    input_position : int
        The position of the input updated.
    replacement_output_leaf : int or None
        The output that holds the updated value; None for an in-place write.
    producer_node : str
        Name of the graph node that makes the update.
    producer_target : str
        The operation that node calls, as text.
    argument_name : str
        What the operation writes: the name of its written argument, or of the
        updated input for a replacement.

    Raises
    ------
    ValueError
        If a position is negative or a name is empty.
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
        """Return the record as plain values, as its contract's digest covers it.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "input_position": self.input_position,
            "replacement_output_leaf": self.replacement_output_leaf,
            "producer_node": self.producer_node,
            "producer_target": self.producer_target,
            "argument_name": self.argument_name,
        }


@dataclass(frozen=True, slots=True)
class TaskStorageContract:
    """What one task promises about the storage of its outputs and its updates.

    A task that does not match its contract is refused, not warned about.
    `TaskStorageContract.build` makes one with its digest computed.

    Parameters
    ----------
    roots : tuple of StorageRoot
        The storages the outputs reach, with root IDs ``0, 1, ...`` in order.
    output_views : tuple of OutputView
        Each output's view, at most one per output. Every root has a view, and
        every view lies inside its root's span.
    mutations : tuple of MutationBinding
        The task's updates to its inputs, none repeated.
    compatibility_digest : str
        SHA-256 of the contract's parts, in hexadecimal, as `digest_of`
        computes it.

    Raises
    ------
    ValueError
        If the root IDs are not contiguous, an output has two views, a view
        names an unknown root or overruns its span, a root has no view, a
        mutation repeats, or the digest is not 64 characters.
    """

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
        """Return the digest these parts have, whether or not they make a contract.

        A record read from a store is checked against its digest before anything
        is built from it, so a changed record is refused as changed rather than for
        whatever its contents then violate.

        Parameters
        ----------
        roots : tuple of StorageRoot
            The roots.
        output_views : tuple of OutputView
            The views.
        mutations : tuple of MutationBinding
            The updates.

        Returns
        -------
        str
            SHA-256 of their sorted compact JSON, in hexadecimal.
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
        """Build a contract over these parts, with the digest they have.

        Parameters
        ----------
        roots : tuple of StorageRoot
            The roots.
        output_views : tuple of OutputView
            The views.
        mutations : tuple of MutationBinding
            The updates.

        Returns
        -------
        TaskStorageContract
            The contract.

        Raises
        ------
        ValueError
            If the parts do not make a valid contract.
        """

        return cls(
            roots=roots,
            output_views=output_views,
            mutations=mutations,
            compatibility_digest=cls.digest_of(roots, output_views, mutations),
        )

    def identity(self) -> dict[str, object]:
        """Return the contract's parts as plain values, as its digest covers them.

        Returns
        -------
        dict
            Its roots, views and mutations.
        """
        return {
            "roots": [root.identity() for root in self.roots],
            "output_views": [view.identity() for view in self.output_views],
            "mutations": [mutation.identity() for mutation in self.mutations],
        }

    def without_device_storage(self, leaves: Collection[int]) -> TaskStorageContract:
        """Return this contract with the named outputs holding no device storage.

        A contract is traced over values that only describe the real ones, and a
        trace can place on the device a result the task really returns elsewhere,
        such as a scalar. No device allocation exists for it, so the contract gives
        way. A named output keeps its identity, shape and element type and gives up
        its span; a root whose outputs are all named gives up its span too, which
        is how the rest of the system says no storage is needed.

        Parameters
        ----------
        leaves : collection of int
            Positions of the outputs observed off the device.

        Returns
        -------
        TaskStorageContract
            The contract with those spans at zero; this one when ``leaves`` is
            empty.

        Raises
        ------
        CaptureError
            If a position names no output, or a root is named for some of its
            outputs but not all of them.
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
        """Return the contract's `to_dict` form as compact sorted JSON.

        Returns
        -------
        str
            The text.
        """

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, object]:
        """Return the contract as JSON values.

        Returns
        -------
        dict
            Its parts, its digest and its schema label.
        """

        return {
            "schema": artifact_schema("task_storage_contract"),
            "compatibility_digest": self.compatibility_digest,
            **self.identity(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TaskStorageContract:
        """Read a contract from its `to_dict` form, checking its digest.

        Parameters
        ----------
        payload : mapping
            The decoded JSON value.

        Returns
        -------
        TaskStorageContract
            The contract.

        Raises
        ------
        ValueError
            If the fields or schema label differ from this version's, a part is
            malformed, or the digest does not match the parts.
        """

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
        """Read a contract from its JSON text.

        Parameters
        ----------
        encoded : str
            The text, as `to_json` writes it.

        Returns
        -------
        TaskStorageContract
            The contract.

        Raises
        ------
        ValueError
            If ``encoded`` is not a JSON object, or as for `from_dict`.
        """

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
