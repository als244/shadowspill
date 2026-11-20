"""Logical input partitions and field delivery, independent of parameter shards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from .groups import GroupRef


@dataclass(frozen=True)
class InputFieldPlacement:
    """How one input field reaches the members of each input partition.

    Parameters
    ----------
    path : sequence of str
        The field's path of keys, such as ``("inputs", "tokens")``. Nonempty
        names; stored as a tuple.
    mode : str, default "replicate"
        ``"replicate"``: every member receives the whole field. ``"slice"``:
        each member receives its own contiguous part along ``axis``, in
        member order; a part may be empty.
    axis : int, optional
        The dimension ``"slice"`` divides: required for it and refused for
        the other modes. Nonnegative.

    Raises
    ------
    ValueError
        If ``path`` is a string or holds an empty name, ``mode`` is another
        value, or ``axis`` is missing for ``"slice"`` or given for another
        mode.
    """

    path: tuple[str, ...]
    mode: Literal["replicate", "slice"] = "replicate"
    axis: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.path, str)
            or not self.path
            or any(not isinstance(part, str) or not part for part in self.path)
        ):
            raise ValueError("input field path must contain nonempty names")
        object.__setattr__(self, "path", tuple(self.path))
        if self.mode not in ("replicate", "slice"):
            raise ValueError("unknown input field delivery mode")
        if self.mode == "slice":
            if type(self.axis) is not int or self.axis < 0:
                raise ValueError("sliced fields require a nonnegative axis")
        elif self.axis is not None:
            raise ValueError("only sliced fields have an axis")

    def region(
        self, shape: Sequence[int], *, index: int, workers: int
    ) -> tuple[slice, ...]:
        """Return the part of the field one member of a partition receives.

        Parameters
        ----------
        shape : sequence of int
            The field's shape in the partition.
        index : int
            The member's position in its partition.
        workers : int
            Members in the partition.

        Returns
        -------
        tuple of slice
            One slice per dimension: the whole dimension, except along
            ``axis`` for ``"slice"``, where the member receives elements
            ``size * index // workers`` up to ``size * (index + 1) // workers``.

        Raises
        ------
        ValueError
            If ``workers`` is not positive, ``index`` is not one of its
            positions, a size is not a nonnegative integer, or ``axis`` is not
            a dimension of ``shape``.
        """
        if (
            type(workers) is not int
            or workers < 1
            or type(index) is not int
            or not 0 <= index < workers
        ):
            raise ValueError("invalid input partition worker index")
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("input shape must contain nonnegative integers")
        regions = [slice(0, size) for size in shape]
        if self.axis is not None:
            if self.axis >= len(shape):
                raise ValueError("input slice axis exceeds the field rank")
            size = shape[self.axis]
            regions[self.axis] = slice(
                size * index // workers, size * (index + 1) // workers
            )
        return tuple(regions)

    def to_dict(self) -> dict[str, Any]:
        """Return the placement as plain data.

        Returns
        -------
        dict
            ``path`` as a list, ``mode`` and ``axis``.
        """
        return {"path": list(self.path), "mode": self.mode, "axis": self.axis}


@dataclass(frozen=True)
class OutputPlacement:
    """How one named output is assembled from the members of each input partition.

    Parameters
    ----------
    mode : str, default "leader"
        ``"leader"``: the partition's first member supplies the output.
        ``"sum"``: the members' outputs are summed. ``"concat"``: each member
        supplies its own contiguous part along ``axis``, joined in member
        order.
    axis : int, optional
        The dimension ``"concat"`` joins along: required for it and refused
        for the other modes. Nonnegative.

    Raises
    ------
    ValueError
        If ``mode`` is another value, or ``axis`` is missing for ``"concat"``
        or given for another mode.
    """

    mode: Literal["leader", "sum", "concat"] = "leader"
    axis: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("leader", "sum", "concat"):
            raise ValueError("unknown output placement mode")
        if self.mode == "concat":
            if type(self.axis) is not int or self.axis < 0:
                raise ValueError("concatenated outputs require a nonnegative axis")
        elif self.axis is not None:
            raise ValueError("only concatenated outputs have an axis")

    def to_dict(self) -> dict[str, Any]:
        """Return the placement as plain data.

        Returns
        -------
        dict
            ``mode`` and ``axis``.
        """
        return {"mode": self.mode, "axis": self.axis}


@dataclass(frozen=True)
class InputPlacement:
    """How each batch's inputs, loss and outputs divide across ranks.

    `Parallelism` takes one as ``inputs``. Each group of ``partitions``
    receives its own part of the batch, and its members cooperate on that
    part. Neither loss rule changes how gradients are reduced or divides by
    a count: the model and objective must compute the share each declares,
    from an empty slice too.

    Parameters
    ----------
    partitions : GroupRef
        The input partitions, one group each, in partition order; a group's
        first member is its leader. `Parallelism` requires them to cover
        every rank exactly once.
    fields : sequence of InputFieldPlacement, default ()
        How each input field reaches a partition's members, at most one entry
        per path. When empty, every field is delivered whole, which run
        preparation refuses when a partition has more than one member.
    loss : str, default "leader"
        Whose loss counts. ``"leader"``: only each partition's leader's, for
        a loss every member computes in full. ``"partitioned"``: every
        member's, summed, for a loss each member computes from its own part.
    outputs : mapping of str to OutputPlacement, default {}
        How each named output is assembled from a partition's members. An
        output may go without an entry only when every partition has one
        member.

    Raises
    ------
    TypeError
        If ``partitions`` is not a `GroupRef`, or an entry of ``fields`` or
        ``outputs`` has another type.
    ValueError
        If two fields share a path, ``loss`` is another value, or an output
        name is empty.
    """

    partitions: GroupRef
    fields: Sequence[InputFieldPlacement] = ()
    loss: Literal["leader", "partitioned"] = "leader"
    outputs: Mapping[str, OutputPlacement] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.partitions, GroupRef):
            raise TypeError("input partitions must be a GroupRef")
        fields = tuple(self.fields)
        if any(not isinstance(item, InputFieldPlacement) for item in fields):
            raise TypeError("input fields require InputFieldPlacement records")
        if len({item.path for item in fields}) != len(fields):
            raise ValueError("input field placement paths must be unique")
        if self.loss not in ("leader", "partitioned"):
            raise ValueError("unknown loss contribution ownership")
        outputs = dict(self.outputs)
        if any(not isinstance(name, str) or not name for name in outputs):
            raise ValueError("output names must be nonempty strings")
        if any(not isinstance(item, OutputPlacement) for item in outputs.values()):
            raise TypeError("output fields require OutputPlacement records")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "outputs", MappingProxyType(outputs))

    @classmethod
    def independent(cls, world_size: int, **options: Any) -> InputPlacement:
        """Declare one input partition per rank.

        Parameters
        ----------
        world_size : int
            Number of ranks. Positive.
        **options
            The other arguments: ``fields``, ``loss`` and ``outputs``.

        Returns
        -------
        InputPlacement
            The placement, whose partitions, named ``"inputs"``, are each
            rank alone in rank order.

        Raises
        ------
        ValueError
            If ``world_size`` is not a positive integer.
        """
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        return cls(
            GroupRef("inputs", tuple((rank,) for rank in range(world_size))), **options
        )

    def validate(self, world_size: int) -> None:
        """Check that the partitions cover every rank exactly once.

        `Parallelism` calls it with its world size.

        Parameters
        ----------
        world_size : int
            Number of ranks: ``0`` to ``world_size - 1``.

        Raises
        ------
        ValueError
            If ``world_size`` is not a positive integer, or the partitions
            miss a rank or hold one outside that range.
        """
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if sorted(rank for group in self.partitions.groups for rank in group) != list(
            range(world_size)
        ):
            raise ValueError(
                "input partitions must cover every logical rank exactly once"
            )

    @property
    def partition_count(self) -> int:
        """Number of input partitions."""
        return len(self.partitions.groups)

    def partition_index(self, logical_rank: int) -> int:
        """Return the input partition a rank belongs to.

        Parameters
        ----------
        logical_rank : int
            The rank.

        Returns
        -------
        int
            The partition's position in ``partitions``.

        Raises
        ------
        ValueError
            If no partition holds the rank.
        """
        members = self.partitions.members(logical_rank)
        return self.partitions.groups.index(members)

    def contributes_loss(self, logical_rank: int) -> bool:
        """Return whether a rank's loss counts toward the step's loss.

        Parameters
        ----------
        logical_rank : int
            The rank.

        Returns
        -------
        bool
            True for every rank under ``"partitioned"``; under ``"leader"``,
            only for a partition's leader.

        Raises
        ------
        ValueError
            If no partition holds the rank.
        """
        index = self.partitions.index(logical_rank)
        return self.loss == "partitioned" or index == 0

    def region(
        self, path: tuple[str, ...], shape: Sequence[int], logical_rank: int
    ) -> tuple[slice, ...]:
        """Return the part of an input field a rank receives.

        Parameters
        ----------
        path : sequence of str
            The field's path.
        shape : sequence of int
            The field's shape in the rank's partition.
        logical_rank : int
            The rank.

        Returns
        -------
        tuple of slice
            One slice per dimension, as `InputFieldPlacement.region`
            describes.

        Raises
        ------
        ValueError
            If ``fields`` has no entry for ``path``, no partition holds the
            rank, or ``shape`` does not suit the field's placement.
        """
        index = self.partitions.index(logical_rank)
        for item in self.fields:
            if item.path == tuple(path):
                return item.region(shape, index=index, workers=self.partitions.size)
        raise ValueError(f"input field has no delivery declaration: {path}")

    def owns_output(self, name: str, logical_rank: int) -> bool:
        """Return whether a rank supplies any of a named output.

        Parameters
        ----------
        name : str
            The output's name, a key of ``outputs``.
        logical_rank : int
            The rank.

        Returns
        -------
        bool
            True for every rank under ``"sum"`` and ``"concat"``; under
            ``"leader"``, only for a partition's leader.

        Raises
        ------
        KeyError
            If ``outputs`` has no entry for ``name``.
        ValueError
            If no partition holds the rank.
        """
        index = self.partitions.index(logical_rank)
        return self.outputs[name].mode != "leader" or index == 0

    def scheduled_capacity(self, capacities: Sequence[Sequence[int]]) -> int:
        """Return the total capacity of the buffers scheduled for the partitions.

        Each partition's buffers count once, however many members share them,
        and each at its full capacity, padding included. Partitions may have
        different numbers of buffers. It checks no schedule.

        Parameters
        ----------
        capacities : sequence of sequence of int
            For each partition, in order, the capacities of its buffers.
            Nonnegative.

        Returns
        -------
        int
            Their sum.

        Raises
        ------
        ValueError
            If ``capacities`` does not have one entry per partition, or a
            capacity is not a nonnegative integer.
        """
        if len(capacities) != self.partition_count:
            raise ValueError("scheduled buffers must describe every input partition")
        if any(
            type(size) is not int or size < 0 for group in capacities for size in group
        ):
            raise ValueError("scheduled buffer capacities must be nonnegative integers")
        return sum(size for group in capacities for size in group)

    def to_dict(self) -> dict[str, Any]:
        """Return the placement as plain data.

        Returns
        -------
        dict
            ``version`` 1; ``partitions`` as its ``name`` and ``groups``; and
            ``fields``, ``loss`` and ``outputs``, each placement as its own
            ``to_dict`` record.
        """
        return {
            "version": 1,
            "partitions": {
                "name": self.partitions.name,
                "groups": [list(group) for group in self.partitions.groups],
            },
            "fields": [item.to_dict() for item in self.fields],
            "loss": self.loss,
            "outputs": {name: item.to_dict() for name, item in self.outputs.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InputPlacement:
        """Rebuild a placement from a `to_dict` record.

        Parameters
        ----------
        value : mapping
            The placement, in `to_dict` form.

        Returns
        -------
        InputPlacement
            The placement.

        Raises
        ------
        ValueError
            If ``version`` is not 1, or the placement is invalid.
        """
        data = dict(value)
        if type(data.get("version")) is not int or data.pop("version") != 1:
            raise ValueError("unsupported input placement version")
        data["partitions"] = GroupRef(**data["partitions"])
        data["fields"] = tuple(InputFieldPlacement(**item) for item in data["fields"])
        data["outputs"] = {
            name: OutputPlacement(**item) for name, item in data["outputs"].items()
        }
        return cls(**data)
