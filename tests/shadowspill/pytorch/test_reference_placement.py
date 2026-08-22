from __future__ import annotations

import random

from reference.python.admission import place_lifetimes as place_reference
from shadowspill.planner._placement import place_lifetimes as place_compiled

_MIB = 1 << 20


def _lifetimes(seed: int, count: int) -> list[tuple[int, int, int, int, int]]:
    """Random leases with the shapes a real schedule produces."""

    rng = random.Random(seed)
    items = []
    for lease_id in range(count):
        start = rng.randrange(0, 10_000)
        items.append(
            (
                rng.choice((1, 4, 512, 4096, 1 << 20)) * rng.randrange(1, 64),
                rng.choice((1, 256, 512, 4096)),
                start,
                start + rng.randrange(0, 3_000),
                lease_id,
            )
        )
    return items


def test_empty_placement_requires_nothing() -> None:
    assert place_compiled([]) == ((), 0)


def test_disjoint_lifetimes_share_one_offset() -> None:
    items = [
        (1024, 512, 0, 10, 0),
        (1024, 512, 10, 20, 1),
        (1024, 512, 20, 30, 2),
    ]
    offsets, required = place_compiled(items)

    assert offsets == (0, 0, 0)
    assert required == 1024


def test_overlapping_lifetimes_stack() -> None:
    items = [
        (1024, 512, 0, 30, 0),
        (1024, 512, 10, 20, 1),
    ]
    offsets, required = place_compiled(items)

    assert offsets == (0, 1024)
    assert required == 2048


def test_alignment_is_honoured() -> None:
    items = [
        (100, 1, 0, 30, 0),
        (100, 512, 0, 30, 1),
    ]
    offsets, required = place_compiled(items)

    assert offsets[1] % 512 == 0
    assert offsets[1] >= 100
    assert required == offsets[1] + 100


def test_result_is_independent_of_input_order() -> None:
    items = _lifetimes(seed=7, count=200)
    shuffled = list(items)
    random.Random(11).shuffle(shuffled)

    _, required = place_compiled(items)
    _, shuffled_required = place_compiled(shuffled)

    assert required == shuffled_required


def test_compiled_placement_matches_readable_reference() -> None:
    for seed in range(12):
        items = _lifetimes(seed=seed, count=150)

        compiled_offsets, compiled_required = place_compiled(items)
        reference_offsets, reference_required = place_reference(items)

        assert compiled_offsets == reference_offsets, f"offsets differ at {seed=}"
        assert compiled_required == reference_required, f"size differs at {seed=}"


def test_placed_leases_never_overlap_in_both_time_and_address() -> None:
    items = _lifetimes(seed=3, count=250)
    offsets, required = place_compiled(items)

    for left in range(len(items)):
        size, _, start, end, _ = items[left]
        for right in range(left + 1, len(items)):
            other_size, _, other_start, other_end, _ = items[right]
            overlaps_in_time = start < other_end and other_start < end
            overlaps_in_address = (
                offsets[left] < offsets[right] + other_size
                and offsets[right] < offsets[left] + size
            )
            assert not (overlaps_in_time and overlaps_in_address)
        assert offsets[left] + size <= required
