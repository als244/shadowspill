"""Prepared tensor-output geometry and inverse maps, without retaining GPU data."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from shadowspill.parallelism import InputPlacement, OutputPlacement
from shadowspill_client._records import plain
from shadowspill_client.inputs.packing import PackedAssignment
from shadowspill_client.inputs.shapes import PackingSchedule
from shadowspill_client.protocol.components import OutputSet, TensorOutput
from shadowspill_client.protocol.inputs import InputSchema, TensorField

from .inputs.materialize import field_regions

type Reduction = Literal["sum", "min", "max", "identical"]


@dataclass(frozen=True)
class OutputDestination:
    name: str
    datum: int | None
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class OutputPiece:
    """One worker view belonging to a caller-ordered tensor destination.

    Combine worker pieces within ``buffer`` first, then apply the separately
    declared global tensor reduction across buffers. This distinction matters
    for a worker sum followed by a microbatch max, for example. Concatenated
    pieces write disjoint destination regions; no concatenation buffer is needed.
    """

    destination: OutputDestination
    buffer: int
    logical_rank: int
    source_shape: tuple[int, ...]
    source_axes: tuple[int | slice, ...]
    destination_axes: tuple[slice, ...]
    worker_reduction: Literal["leader", "sum", "concat"]
    global_reduction: Reduction | None


def _axis(field: TensorField, dimension: str) -> int:
    if field.pack_axis is not None and field.shape[field.pack_axis] == dimension:
        return field.pack_axis
    return field.shape.index(dimension)


class OutputLayout:
    """Resolve declared alignment once; execution uses exact saved coordinates."""

    def __init__(
        self,
        schema: InputSchema,
        outputs: OutputSet,
        schedule: PackingSchedule,
        placement: InputPlacement,
    ) -> None:
        self.schema, self.outputs = schema, outputs
        self.schedule, self.placement = schedule, placement
        if len(schedule.partitions) != placement.partition_count:
            raise ValueError("output schedule must describe every input partition")
        definitions = {tuple(field.path): field for field in schema.fields}
        for buffer in schedule.buffers:
            buffer.validate(definitions)
        selected = {
            name: tuple(output.source)
            for name, output in outputs.tensors.items()
            if output.source is not None
        }
        aligned = {
            name
            for name, output in outputs.tensors.items()
            if output.ordering == "aligned"
            or any(isinstance(d, str) for d in output.shape)
        }
        if selected.keys() - aligned:
            raise ValueError("only symbolic tensor outputs name input sources")
        reductions = {
            name: output.reduction
            for name, output in outputs.tensors.items()
            if output.reduction is not None
        }
        for name, output in outputs.tensors.items():
            if output.ordering == "global":
                if any(type(d) is not int for d in output.shape):
                    raise ValueError(
                        "global tensor shapes must have resolved fixed dimensions"
                    )
                continue
            if name not in aligned:
                continue
            candidates = [
                path
                for path, field in definitions.items()
                if path in schedule.buffers[0].fields
                and all(d in field.shape for d in output.shape if isinstance(d, str))
                and (
                    output.ordering == "datum"
                    or field.batching == "stack"
                    or (
                        field.pack_axis is not None
                        and field.shape[field.pack_axis] == output.alignment
                    )
                )
            ]
            if name in selected:
                if selected[name] not in candidates:
                    raise ValueError(
                        f"output {name} has an incompatible alignment source"
                    )
            elif (
                not candidates
                or len(
                    {
                        self._alignment_identity(definitions[path], output)
                        for path in candidates
                    }
                )
                != 1
            ):
                raise ValueError(
                    f"output {name} requires an explicit unambiguous alignment source"
                )
            else:
                selected[name] = sorted(candidates)[0]
        self.sources = MappingProxyType(selected)
        self.global_reductions = MappingProxyType(reductions)
        self._definitions = definitions
        for name in outputs.tensors:
            rule = self._rule(name)
            for index in range(len(schedule.buffers)):
                shape = self.physical_shape(name, index)
                if rule.axis is not None and rule.axis >= len(shape):
                    raise ValueError(
                        f"output {name} concatenation axis exceeds its rank"
                    )

    def _alignment_identity(self, field: TensorField, output: TensorOutput) -> object:
        path = tuple(field.path)
        stacked = field.batching == "stack"
        # Equivalent declared symbols/packing axes have identical inverse regions;
        # equal numeric sizes without those symbols never establish alignment.
        return (
            stacked if output.ordering == "aligned" else True,
            tuple(
                (_axis(field, d) == field.pack_axis and output.ordering == "aligned")
                if isinstance(d, str)
                else None
                for d in output.shape
            ),
            tuple(
                tuple(
                    buffer.fields[path][_axis(field, d) + stacked]
                    if isinstance(d, str)
                    else d
                    for d in output.shape
                )
                for buffer in self.schedule.buffers
            ),
        )

    def _rule(self, name: str) -> OutputPlacement:
        rule = self.placement.outputs.get(name)
        if rule is None:
            if any(len(group) != 1 for group in self.placement.partitions.groups):
                raise ValueError(
                    f"output {name} needs explicit cooperating-worker placement"
                )
            return OutputPlacement()
        return rule

    def physical_shape(self, name: str, buffer: int) -> tuple[int, ...]:
        output = self.outputs.tensors[name]
        shape = self.schedule.buffers[buffer]
        if name not in self.sources:
            fixed = tuple(int(n) for n in output.shape)
            return (shape.max_datums, *fixed) if output.ordering == "datum" else fixed
        path = self.sources[name]
        field = self._definitions[path]
        stacked = field.batching == "stack"
        dimensions = tuple(
            shape.fields[path][_axis(field, d) + stacked] if isinstance(d, str) else d
            for d in output.shape
        )
        return (
            (shape.max_datums, *dimensions)
            if stacked or output.ordering == "datum"
            else dimensions
        )

    def worker_shape(
        self, name: str, buffer: int, logical_rank: int
    ) -> tuple[int, ...]:
        part, _ = self.schedule.locations[buffer]
        group = self.placement.partitions.groups[part]
        if logical_rank not in group:
            raise ValueError("output worker does not own this input partition")
        shape = list(self.physical_shape(name, buffer))
        rule = self._rule(name)
        if rule.mode == "concat":
            assert rule.axis is not None
            length, position = shape[rule.axis], group.index(logical_rank)
            shape[rule.axis] = length * (position + 1) // len(
                group
            ) - length * position // len(group)
        return tuple(shape)

    def destinations(self, assignment: PackedAssignment) -> Iterator[OutputDestination]:
        self._check(assignment)
        for name, output in self.outputs.tensors.items():
            if output.ordering == "global":
                yield OutputDestination(
                    name, None, output.dtype, tuple(int(n) for n in output.shape)
                )
            else:
                for index, dimensions in enumerate(assignment.batch.dimensions):
                    yield OutputDestination(
                        name,
                        index,
                        output.dtype,
                        tuple(
                            dimensions[d] if isinstance(d, str) else d
                            for d in output.shape
                        ),
                    )

    def _check(self, assignment: PackedAssignment) -> None:
        if assignment.batch.owner.closed:
            raise RuntimeError("output mapping requires its owned input metadata")
        if (
            assignment.batch.schema != self.schema
            or assignment.schedule != self.schedule
        ):
            raise ValueError("output layout differs from the packed input contract")

    def pieces(self, assignment: PackedAssignment) -> Iterator[OutputPiece]:
        self._check(assignment)
        for buffer, (part, _) in enumerate(self.schedule.locations):
            regions = field_regions(assignment, self.schema, buffer)
            for name, output in self.outputs.tensors.items():
                full = self.physical_shape(name, buffer)
                entries: list[tuple[int | None, tuple[int | slice, ...]]]
                if output.ordering == "global":
                    entries = [(None, tuple(slice(0, n) for n in full))]
                elif output.ordering == "datum":
                    entries = [
                        (
                            region.datum,
                            (
                                region.slot,
                                *(
                                    slice(
                                        0,
                                        assignment.batch.dimensions[region.datum][n]
                                        if isinstance(n, str)
                                        else n,
                                    )
                                    for n in output.shape
                                ),
                            ),
                        )
                        for region in regions
                    ]
                else:
                    path = self.sources[name]
                    field = self._definitions[path]
                    stacked = field.batching == "stack"
                    entries = [
                        (
                            region.datum,
                            (
                                *((region.slot,) if stacked else ()),
                                *(
                                    region.fields[path][_axis(field, d) + stacked]
                                    if isinstance(d, str)
                                    else slice(0, d)
                                    for d in output.shape
                                ),
                            ),
                        )
                        for region in regions
                    ]
                rule = self._rule(name)
                group = self.placement.partitions.groups[part]
                ranks = group[:1] if rule.mode == "leader" else group
                for datum, axes in entries:
                    logical = tuple(
                        axis.stop - axis.start
                        for axis in axes
                        if isinstance(axis, slice)
                    )
                    destination = OutputDestination(name, datum, output.dtype, logical)
                    for rank in ranks:
                        source_axes = list(axes)
                        target_axes = [slice(0, n) for n in logical]
                        if rule.mode == "concat":
                            assert rule.axis is not None
                            axis, index = rule.axis, group.index(rank)
                            start = full[axis] * index // len(group)
                            end = full[axis] * (index + 1) // len(group)
                            selection = axes[axis]
                            if isinstance(selection, int):
                                if not start <= selection < end:
                                    continue
                                source_axes[axis] = selection - start
                            else:
                                left, right = (
                                    max(start, selection.start),
                                    min(end, selection.stop),
                                )
                                if left > right or (
                                    left == right and selection.stop != selection.start
                                ):
                                    continue
                                source_axes[axis] = slice(left - start, right - start)
                                destination_axis = sum(
                                    isinstance(a, slice) for a in axes[:axis]
                                )
                                target_axes[destination_axis] = slice(
                                    left - selection.start, right - selection.start
                                )
                        yield OutputPiece(
                            destination,
                            buffer,
                            rank,
                            self.worker_shape(name, buffer, rank),
                            tuple(source_axes),
                            tuple(target_axes),
                            rule.mode,
                            self.global_reductions.get(name),
                        )

    def metric_workers(self, buffer: int) -> tuple[int, ...]:
        """Scalar reductions count each declared logical contribution once."""
        part, _ = self.schedule.locations[buffer]
        return tuple(
            rank
            for rank in self.placement.partitions.groups[part]
            if self.placement.contributes_loss(rank)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "schema": self.schema.to_dict(),
            "outputs": self.outputs.to_dict(),
            "schedule": self.schedule.to_dict(),
            "placement": self.placement.to_dict(),
            "sources": {name: list(path) for name, path in self.sources.items()},
            "global_reductions": dict(self.global_reductions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OutputLayout:
        if (
            type(value.get("version")) is not int
            or value["version"] != 1
            or set(value)
            != {
                "version",
                "schema",
                "outputs",
                "schedule",
                "placement",
                "sources",
                "global_reductions",
            }
        ):
            raise ValueError("unsupported output layout version")
        result = cls(
            InputSchema.from_dict(value["schema"]),
            OutputSet.from_dict(value["outputs"]),
            PackingSchedule.from_dict(value["schedule"]),
            InputPlacement.from_dict(value["placement"]),
        )
        if result.to_dict() != plain(value):
            raise ValueError("saved output layout differs from its declared contract")
        return result
