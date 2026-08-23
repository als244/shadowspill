from __future__ import annotations

import random
from dataclasses import dataclass

from reference.python.admission import place_lifetimes as place_reference
from shadowspill.planner._placement import place_lifetimes as place


@dataclass(frozen=True, slots=True)
class _Lease:
    """The four numbers placement reads, and nothing else."""

    bytes: int
    alignment: int
    predicted_start_ns: int
    predicted_end_ns: int


def _lifetimes(seed: int, count: int) -> list[_Lease]:
    """Random leases with the shapes a real schedule produces."""

    rng = random.Random(seed)
    items = []
    for _ in range(count):
        start = rng.randrange(0, 10_000)
        items.append(
            _Lease(
                bytes=rng.choice((1, 4, 512, 4096, 1 << 20)) * rng.randrange(1, 64),
                alignment=rng.choice((1, 256, 512, 4096)),
                predicted_start_ns=start,
                predicted_end_ns=start + rng.randrange(0, 3_000),
            )
        )
    return items


def _distinct_sizes(seed: int, count: int) -> list[_Lease]:
    """Leases no two of which can tie, so the order is fully determined."""

    rng = random.Random(seed)
    return [
        _Lease(
            bytes=(index + 1) * 4096,
            alignment=rng.choice((1, 256, 512)),
            predicted_start_ns=(start := rng.randrange(0, 10_000)),
            predicted_end_ns=start + rng.randrange(0, 3_000),
        )
        for index in range(count)
    ]


def test_empty_placement_requires_nothing() -> None:
    assert place([]) == ((), 0)


def test_disjoint_lifetimes_share_one_offset() -> None:
    items = [
        _Lease(1024, 512, 0, 10),
        _Lease(1024, 512, 10, 20),
        _Lease(1024, 512, 20, 30),
    ]
    offsets, required = place(items)

    assert offsets == (0, 0, 0)
    assert required == 1024


def test_overlapping_lifetimes_stack() -> None:
    items = [
        _Lease(1024, 512, 0, 30),
        _Lease(1024, 512, 10, 20),
    ]
    offsets, required = place(items)

    assert offsets == (0, 1024)
    assert required == 2048


def test_alignment_is_honoured() -> None:
    items = [
        _Lease(100, 1, 0, 30),
        _Lease(100, 512, 0, 30),
    ]
    offsets, required = place(items)

    assert offsets[1] % 512 == 0
    assert offsets[1] >= 100
    assert required == offsets[1] + 100


def test_result_is_independent_of_input_order() -> None:
    """Placement has no identity column, so the input index breaks ties; where
    nothing ties, the layout is a function of the records alone."""

    items = _distinct_sizes(seed=7, count=200)
    shuffled = list(items)
    random.Random(11).shuffle(shuffled)

    _, required = place(items)
    _, shuffled_required = place(shuffled)

    assert required == shuffled_required


def test_compiled_placement_matches_readable_reference() -> None:
    for seed in range(12):
        items = _lifetimes(seed=seed, count=150)

        compiled_offsets, compiled_required = place(items)
        reference_offsets, reference_required = place_reference(items)

        assert compiled_offsets == reference_offsets, f"offsets differ at {seed=}"
        assert compiled_required == reference_required, f"size differs at {seed=}"


def test_placed_leases_never_overlap_in_both_time_and_address() -> None:
    items = _lifetimes(seed=3, count=250)
    offsets, required = place(items)

    for left, item in enumerate(items):
        for right in range(left + 1, len(items)):
            other = items[right]
            overlaps_in_time = (
                item.predicted_start_ns < other.predicted_end_ns
                and other.predicted_start_ns < item.predicted_end_ns
            )
            overlaps_in_address = (
                offsets[left] < offsets[right] + other.bytes
                and offsets[right] < offsets[left] + item.bytes
            )
            assert not (overlaps_in_time and overlaps_in_address)
        assert offsets[left] + item.bytes <= required
