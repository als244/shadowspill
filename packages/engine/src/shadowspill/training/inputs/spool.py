"""Finish every CPU adapter once before exposing a request for execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Self
from uuid import uuid4

from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.packing import PackedAssignment
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.payload import BatchSnapshot
from shadowspill_client.protocol.payload_io import read_payload
from shadowspill_client.protocol.requests import ArtifactRef

from .adapters import InputBinding
from .frames import InputFrame
from .materialize import materialize_buffer
from .routing import route_buffer


@dataclass(frozen=True)
class SpooledFrame:
    path: Path
    binding: InputBinding
    reference: ArtifactRef

    @property
    def layout(self) -> InputFrame:
        return self.binding.frame

    def read(self, budget: BufferBudget) -> BatchSnapshot:
        if self.path.stat().st_size != self.reference.size_bytes:
            raise ProtocolError("prepared input file differs from its declared size")
        with self.path.open("rb", buffering=0) as stream:
            result = read_payload(
                stream,
                self.reference,
                budget,
                timeout=0,
                limits=self.layout.size,
                datum_count=1,
            )
        if result.header != self.layout.header:
            result.close()
            raise ProtocolError("prepared input file differs from its declared layout")
        return result


class InputSpool:
    """Ephemeral, bounded files owned until worker input consumers finish.

    No uploaded code or pickled values enter these files. All data-dependent CPU
    callbacks have finished before construction returns. A worker reads checked
    ready frames; it never reruns preprocessing after a previous microbatch.
    Files do not imply crash-transparent execution or replay after server loss.
    """

    def __init__(
        self,
        assignment: PackedAssignment,
        bindings: Mapping[tuple[int, int], InputBinding],
        budget: BufferBudget,
        *,
        root: Path,
    ) -> None:
        if not bindings:
            raise ValueError("input spooling requires prepared worker bindings")
        first = next(iter(bindings.values())).layout
        placement, schema = first.placement, first.schema
        expected = {
            (buffer, rank)
            for buffer, (partition, _) in enumerate(assignment.schedule.locations)
            for rank in placement.partitions.groups[partition]
        }
        if bindings.keys() != expected or any(
            binding.layout.placement != placement
            or binding.layout.schema != schema
            or binding.layout.logical_rank != rank
            or binding.layout.buffer != assignment.schedule.buffers[buffer]
            for (buffer, rank), binding in bindings.items()
        ):
            raise ValueError(
                "spool bindings differ from the complete prepared schedule"
            )
        # The caller reserves complete validation/packing/frame bookkeeping for
        # this finite schedule before invoking this constructor.
        self._directory = TemporaryDirectory(prefix="inputs-", dir=root)
        self.frames: Mapping[tuple[int, int], SpooledFrame] = MappingProxyType({})
        self.closed = False
        try:
            frames = {}
            for buffer, (partition, _) in enumerate(assignment.schedule.locations):
                with materialize_buffer(assignment, schema, buffer, budget) as packed:
                    for rank in placement.partitions.groups[partition]:
                        binding = bindings[buffer, rank]
                        with binding.adapt(
                            route_buffer(packed, placement, rank), budget
                        ) as adapted:
                            name = uuid4().hex
                            path = Path(self._directory.name) / name
                            with path.open("xb", buffering=0) as stream:
                                reference = binding.frame.write(
                                    stream, binding.frame_fields(adapted), budget, name
                                )
                            frames[buffer, rank] = SpooledFrame(
                                path, binding, reference
                            )
            self.frames = MappingProxyType(frames)
        except BaseException:
            self.close()
            raise

    @property
    def size_bytes(self) -> int:
        return sum(frame.reference.size_bytes for frame in self.frames.values())

    def close(self) -> None:
        self._directory.cleanup()
        self.frames = MappingProxyType({})
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
