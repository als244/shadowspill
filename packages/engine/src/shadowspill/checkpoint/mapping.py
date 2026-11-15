"""Bounded intersections of explicit logical views and owned archive regions."""

from collections.abc import Iterator
from dataclasses import dataclass
from math import prod

from shadowspill.artifacts._values import nonnegative_int
from shadowspill.artifacts.tensor_spans import spans


@dataclass(frozen=True)
class LogicalSpan:
    logical_offset: int
    archive_offset: int
    elements: int


@dataclass(frozen=True)
class LogicalView:
    """A rectangular logical slice over strided, partially owned storage.

    All offsets and lengths are dtype elements, not bytes. ``origin`` addresses
    the global logical tensor; ``storage_offset`` addresses the rank-local
    storage view. An archive contains only ``region_elements`` beginning at
    ``region_offset``. These coordinates are explicit: rank IDs, parameter names
    and equal byte counts do not establish a mapping.
    """

    shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    origin: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage_elements: int
    region_offset: int
    region_elements: int

    def __post_init__(self) -> None:
        for name in ("shape", "local_shape", "origin", "stride"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise TypeError(f"logical {name} must be an integer sequence")
            for value in values:
                nonnegative_int(value, name)
            object.__setattr__(self, name, tuple(values))
        if (
            len(
                {
                    len(self.shape),
                    len(self.local_shape),
                    len(self.origin),
                    len(self.stride),
                }
            )
            != 1
        ):
            raise ValueError("logical view dimensions differ")
        for extent, size, start in zip(
            self.shape, self.local_shape, self.origin, strict=True
        ):
            if start + size > extent:
                raise ValueError("local logical view exceeds the global shape")
        for name in (
            "storage_offset",
            "storage_elements",
            "region_offset",
            "region_elements",
        ):
            nonnegative_int(getattr(self, name), name)
        if self.region_offset + self.region_elements > self.storage_elements:
            raise ValueError("owned region exceeds its storage")
        extent = (
            self.storage_offset
            + 1
            + sum(
                (size - 1) * step
                for size, step in zip(self.local_shape, self.stride, strict=True)
            )
        )
        if prod(self.local_shape) and extent > self.storage_elements:
            raise ValueError("strided logical view exceeds its storage")

    def spans(self) -> Iterator[LogicalSpan]:
        """Coalesce adjacent logical/file ranges while retaining at most one span."""
        pending = None
        for span in self._spans():
            if pending is not None and (
                pending.logical_offset + pending.elements == span.logical_offset
                and pending.archive_offset + pending.elements == span.archive_offset
            ):
                pending = LogicalSpan(
                    pending.logical_offset,
                    pending.archive_offset,
                    pending.elements + span.elements,
                )
            else:
                if pending is not None:
                    yield pending
                pending = span
        if pending is not None:
            yield pending

    def _spans(self) -> Iterator[LogicalSpan]:
        global_stride = tuple(
            prod(self.shape[index + 1 :]) for index in range(len(self.shape))
        )
        origin = sum(
            start * step for start, step in zip(self.origin, global_stride, strict=True)
        )
        physical = iter(spans(self.local_shape, self.stride, self.storage_offset))
        logical = iter(spans(self.local_shape, global_stride, origin))
        left, right = next(physical, None), next(logical, None)
        while left is not None and right is not None:
            begin = max(left[0], right[0])
            end = min(left[0] + left[2], right[0] + right[2])
            storage = left[1] + begin - left[0]
            selected = max(storage, self.region_offset)
            stop = min(storage + end - begin, self.region_offset + self.region_elements)
            if selected < stop:
                yield LogicalSpan(
                    right[1] + begin - right[0] + selected - storage,
                    selected - self.region_offset,
                    stop - selected,
                )
            if end == left[0] + left[2]:
                left = next(physical, None)
            if end == right[0] + right[2]:
                right = next(logical, None)


@dataclass(frozen=True)
class Intersection:
    logical_offset: int
    source_offset: int
    target_offset: int
    elements: int


def intersections(source: LogicalView, target: LogicalView) -> Iterator[Intersection]:
    """Stream one pair's common coordinates; the owner validates global coverage.

    Replication policy, tensor identities, dtype/encoding and tie compatibility
    belong to the global inventory. This helper never selects or sums replicas,
    allocates payloads or assumes that one source covers the complete target.
    """
    if source.shape != target.shape:
        raise ValueError("logical source and target shapes differ")
    sources, targets = source.spans(), target.spans()
    left, right = next(sources, None), next(targets, None)
    while left is not None and right is not None:
        begin = max(left.logical_offset, right.logical_offset)
        end = min(
            left.logical_offset + left.elements, right.logical_offset + right.elements
        )
        if begin < end:
            yield Intersection(
                begin,
                left.archive_offset + begin - left.logical_offset,
                right.archive_offset + begin - right.logical_offset,
                end - begin,
            )
        if left.logical_offset + left.elements <= right.logical_offset + right.elements:
            left = next(sources, None)
        else:
            right = next(targets, None)
