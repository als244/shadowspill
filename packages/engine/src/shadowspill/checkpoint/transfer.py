"""Copy one logical state tensor between owned views with bounded CPU staging."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import prod
from typing import Literal

import torch

from shadowspill.artifacts.tensors import TensorChunk

from .mapping import LogicalView, coverage
from .reading import RankReader


@dataclass(frozen=True)
class SourceView:
    reader: RankReader
    chunk: TensorChunk
    view: LogicalView

    def __post_init__(self) -> None:
        if self.reader.chunks.get(self.chunk.name) != self.chunk:
            raise ValueError("logical source does not name a verified stored tensor")
        elements = (
            self.chunk.storage_elements
            if self.chunk.region_elements is None
            else self.chunk.region_elements
        )
        if (
            self.view.region_offset,
            self.view.region_elements,
            self.view.storage_elements,
        ) != (self.chunk.region_offset, elements, self.chunk.storage_elements):
            raise ValueError("logical source ownership differs from its archive")


@dataclass(frozen=True)
class TargetView:
    view: LogicalView
    dtype: torch.dtype
    write: Callable[[int, torch.Tensor], None]


def transfer_tensor(
    sources: Sequence[SourceView],
    targets: Sequence[TargetView],
    *,
    scratch_bytes: int,
    transformation: Literal["identity", "scalar_cast"] = "identity",
) -> None:
    """Validate coverage and replicas, then stream into unpublished destinations.

    Callbacks consume read-only borrowed CPU buffers before returning. Offsets
    address the target's owned archive region. They must account for any staging
    they retain separately and must not publish partially written destinations.
    State identity, alias relationships and permission to cast authoritative
    ordinary weight representations are checked by the conversion owner.

    Every source replica is compared bit-for-bit before its range is written;
    replicas are never summed or averaged. A later inconsistency or write failure
    aborts conversion. The caller discards all unpublished target output.
    """
    if not sources or not targets:
        raise ValueError("logical transfer requires source and target views")
    source_dtype, target_dtype = sources[0].chunk.dtype, targets[0].dtype
    if any(source.chunk.dtype != source_dtype for source in sources) or any(
        target.dtype != target_dtype for target in targets
    ):
        raise ValueError("logical state copies disagree on dtype")
    if transformation == "identity":
        if source_dtype != target_dtype:
            raise ValueError(
                "logical state dtype change needs an explicit transformation"
            )
    elif transformation != "scalar_cast" or not (
        source_dtype.is_floating_point and target_dtype.is_floating_point
    ):
        raise ValueError("unsupported logical state transformation")
    source_views = [source.view for source in sources]
    target_views = [target.view for target in targets]
    if source_views[0].shape != target_views[0].shape:
        raise ValueError("logical source and target shapes differ")
    # Complete both metadata checks before the first read or destination write.
    for views in (source_views, target_views):
        for _ in coverage(views):
            pass
    element_bytes = 2 * source_dtype.itemsize + target_dtype.itemsize
    if type(scratch_bytes) is not int or scratch_bytes < element_bytes:
        raise ValueError(
            "logical transfer scratch cannot hold one comparison and output"
        )
    width = min(
        scratch_bytes // element_bytes,
        *(source.reader.chunk_bytes // source_dtype.itemsize for source in sources),
    )
    if not width:
        raise ValueError("logical source read budget cannot hold one element")
    source_ranges, target_ranges = coverage(source_views), coverage(target_views)
    left, right = next(source_ranges, None), next(target_ranges, None)
    if left is None:
        return  # Empty logical tensors have no payload to stage or publish.
    width = min(width, prod(source_views[0].shape))
    primary = torch.empty(width, dtype=source_dtype, device="cpu")
    replica = torch.empty_like(primary)
    output = torch.empty(width, dtype=target_dtype, device="cpu")
    while left is not None and right is not None:
        begin = max(left.logical_offset, right.logical_offset)
        end = min(
            left.logical_offset + left.elements, right.logical_offset + right.elements
        )
        for start in range(begin, end, width):
            count = min(width, end - start)
            selected = primary[:count]
            for index, copy in enumerate(left.copies):
                source = sources[copy.view]
                destination = selected if not index else replica[:count]
                source.reader.readinto(
                    replace(
                        source.chunk,
                        offset=source.chunk.region_offset
                        + copy.archive_offset
                        + start
                        - left.logical_offset,
                    ),
                    destination,
                )
                if index and not torch.equal(
                    selected.view(torch.uint8), destination.view(torch.uint8)
                ):
                    raise ValueError(
                        f"logical source replicas disagree at element {start}"
                    )
            converted = output[:count]
            converted.copy_(selected)
            for copy in right.copies:
                targets[copy.view].write(
                    copy.archive_offset + start - right.logical_offset, converted
                )
        if left.logical_offset + left.elements <= right.logical_offset + right.elements:
            left = next(source_ranges, None)
        else:
            right = next(target_ranges, None)
