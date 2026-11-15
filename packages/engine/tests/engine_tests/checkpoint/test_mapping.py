"""Logical coordinate mapping checked against independent NumPy views."""

from dataclasses import replace
from math import prod

import numpy as np
import pytest

from shadowspill.checkpoint.mapping import (
    LogicalSpan,
    LogicalView,
    coverage,
    intersections,
)


def view(
    shape, *, local=None, origin=None, stride=None, offset=0, storage=None, region=None
):
    local = shape if local is None else local
    origin = (0,) * len(shape) if origin is None else origin
    stride = (
        tuple(prod(local[i + 1 :]) for i in range(len(local)))
        if stride is None
        else stride
    )
    storage = prod(local) if storage is None else storage
    region = (0, storage) if region is None else region
    return LogicalView(shape, local, origin, stride, offset, storage, *region)


def expected(mapping):
    global_ids = np.arange(prod(mapping.shape)).reshape(mapping.shape)
    selected = global_ids[
        tuple(
            slice(start, start + size)
            for start, size in zip(mapping.origin, mapping.local_shape, strict=True)
        )
    ]
    storage = np.arange(mapping.storage_elements, dtype=np.int64)
    physical = np.ndarray(
        mapping.local_shape,
        dtype=np.int64,
        buffer=storage,
        offset=mapping.storage_offset * 8,
        strides=tuple(step * 8 for step in mapping.stride),
    )
    return {
        int(logical): int(position) - mapping.region_offset
        for logical, position in zip(selected.flat, physical.flat, strict=True)
        if mapping.region_offset
        <= position
        < mapping.region_offset + mapping.region_elements
    }


@pytest.mark.parametrize(
    "mapping",
    [
        view((6, 8), local=(3, 8), origin=(3, 0)),
        view((6, 8), local=(6, 2), origin=(0, 4), region=(3, 8)),
        view((4, 3), stride=(1, 4)),
        view((2, 3), stride=(9, 2), offset=1, storage=20, region=(3, 11)),
        view((5,), storage=8, region=(4, 4)),
        view((), region=(0, 1)),
        view((), region=(0, 0)),
        view((0, 3)),
        view((3,), stride=(0,), storage=1),
        view(
            (4, 5, 6),
            local=(2, 3, 2),
            origin=(1, 1, 3),
            stride=(1, 4, 12),
            storage=24,
            region=(6, 12),
        ),
    ],
)
def test_owned_logical_spans_match_numpy_storage_views(mapping):
    observed = {}
    for span in mapping.spans():
        assert span.elements > 0
        assert (
            0
            <= span.archive_offset
            < span.archive_offset + span.elements
            <= mapping.region_elements
        )
        for index in range(span.elements):
            assert span.logical_offset + index not in observed
            observed[span.logical_offset + index] = span.archive_offset + index
    assert observed == expected(mapping)


def test_two_row_shards_to_four_column_shards_preserve_each_element():
    original = np.arange(48, dtype=np.float32).reshape(6, 8)
    sources = [view((6, 8), local=(3, 8), origin=(start, 0)) for start in (0, 3)]
    for column in range(4):
        target = view((6, 8), local=(6, 2), origin=(0, 2 * column))
        output = np.full(12, np.nan, dtype=np.float32)
        written = np.zeros(12, dtype=np.int64)
        for rank, source in enumerate(sources):
            payload = original[3 * rank : 3 * rank + 3].reshape(-1)
            for part in intersections(source, target):
                source_slice = slice(
                    part.source_offset, part.source_offset + part.elements
                )
                target_slice = slice(
                    part.target_offset, part.target_offset + part.elements
                )
                output[target_slice] = payload[source_slice]
                written[target_slice] += 1
        np.testing.assert_array_equal(
            output.reshape(6, 2), original[:, 2 * column : 2 * column + 2]
        )
        np.testing.assert_array_equal(written, np.ones(12))


@pytest.mark.parametrize(
    "source_count,target_count", [(1, 4), (4, 1), (2, 3), (3, 2), (4, 4)]
)
def test_partial_flat_storage_shards_convert_without_copying_padding(
    source_count, target_count
):
    original = np.arange(15)

    def shard(rank, count):
        capacity = (15 + count - 1) // count
        start = min(rank * capacity, 15)
        return view((3, 5), region=(start, min(capacity, 15 - start)))

    for rank in range(target_count):
        target = shard(rank, target_count)
        output = np.full(target.region_elements, -1)
        for source_rank in range(source_count):
            source = shard(source_rank, source_count)
            payload = original[
                source.region_offset : source.region_offset + source.region_elements
            ]
            for part in intersections(source, target):
                output[part.target_offset : part.target_offset + part.elements] = (
                    payload[part.source_offset : part.source_offset + part.elements]
                )
        np.testing.assert_array_equal(
            output,
            original[
                target.region_offset : target.region_offset + target.region_elements
            ],
        )


def test_transposed_and_disjoint_logical_views_use_coordinates_not_equal_sizes():
    source = view((4, 4), stride=(1, 4), region=(2, 9))
    target = view((4, 4), local=(2, 4), origin=(1, 0))
    left, right = expected(source), expected(target)
    actual = {}
    for part in intersections(source, target):
        actual.update(
            {
                part.logical_offset + index: (
                    part.source_offset + index,
                    part.target_offset + index,
                )
                for index in range(part.elements)
            }
        )
    assert actual == {
        key: (left[key], right[key]) for key in left.keys() & right.keys()
    }
    assert not list(
        intersections(view((4,), local=(2,)), view((4,), local=(2,), origin=(2,)))
    )
    with pytest.raises(ValueError, match="shapes differ"):
        list(intersections(view((4,)), view((2, 2))))


def test_huge_contiguous_views_stay_a_single_span_without_payload_allocation():
    count = 25_000_000_000
    whole = view((count,))
    assert list(whole.spans()) == [LogicalSpan(0, 0, count)]
    parts = list(
        intersections(whole, replace(whole, region_offset=9, region_elements=count - 9))
    )
    assert len(parts) == 1 and parts[0].elements == count - 9


@pytest.mark.parametrize(
    "change",
    [
        {"shape": (True,)},
        {"local_shape": (2, 2)},
        {"origin": (3,)},
        {"stride": (-1,)},
        {"region_offset": 2, "region_elements": 3},
        {"storage_elements": 3},
        {"storage_offset": 1},
    ],
)
def test_invalid_view_is_rejected_without_iterating_payloads(change):
    with pytest.raises((TypeError, ValueError)):
        replace(view((4,)), **change)


def test_coverage_enumerates_every_copy_including_strides_and_partial_overlap():
    views = [
        view((3, 4), region=(0, 7)),
        view((3, 4), region=(7, 5)),
        view((3, 4), stride=(1, 3)),
        view((3, 4), local=(2, 2), origin=(1, 1)),
    ]
    expected_views = [expected(item) for item in views]
    seen = {}
    for region in coverage(views):
        for offset in range(region.elements):
            logical = region.logical_offset + offset
            assert logical not in seen
            seen[logical] = [
                (copy.view, copy.archive_offset + offset) for copy in region.copies
            ]
    assert seen == {
        element: [
            (index, item[element])
            for index, item in enumerate(expected_views)
            if element in item
        ]
        for element in range(12)
    }


@pytest.mark.parametrize(
    "regions,gap", [([(1, 4)], 0), ([(0, 2), (3, 2)], 2), ([(0, 4)], 4)]
)
def test_coverage_rejects_missing_prefix_middle_or_tail(regions, gap):
    with pytest.raises(ValueError, match=f"gap at element {gap}"):
        list(coverage([view((5,), region=region) for region in regions]))


def test_coverage_handles_empty_and_huge_replicated_views_without_payloads():
    assert not list(coverage([view((0, 4)), view((0, 4))]))
    regions = list(coverage([view((25_000_000_000,))] * 4))
    assert len(regions) == 1 and len(regions[0].copies) == 4
    assert regions[0].elements == 25_000_000_000
    with pytest.raises(ValueError, match="at least one"):
        list(coverage([]))
    with pytest.raises(ValueError, match="shapes differ"):
        list(coverage([view((4,)), view((2, 2))]))
