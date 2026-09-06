from __future__ import annotations

import pytest

from shadowspill.planner import StepDataOrdering


def test_resolve_fills_in_the_count_left_out_and_defaults_to_depth_first() -> None:
    assert StepDataOrdering.resolve(microbatches=8) == StepDataOrdering(8, 1)
    assert StepDataOrdering.resolve(microbatches=8, breadth=4) == StepDataOrdering(2, 4)
    assert StepDataOrdering.resolve(microbatches=8, depth=2) == StepDataOrdering(2, 4)
    assert StepDataOrdering.resolve(
        microbatches=8, depth=2, breadth=4, reverse_breadth=False, pair_loss=False
    ) == StepDataOrdering(2, 4, False, False)


@pytest.mark.parametrize(
    ("depth", "breadth", "message"),
    [
        (3, 4, "3 times breadth 4 is 12, but the step has 8"),
        (None, 3, "breadth 3 does not divide the 8"),
        (5, None, "depth 5 does not divide the 8"),
    ],
)
def test_resolve_names_every_number_in_a_mismatch(
    depth: int | None, breadth: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StepDataOrdering.resolve(microbatches=8, depth=depth, breadth=breadth)


def test_counts_must_be_positive_integers() -> None:
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        StepDataOrdering(0, 4)
    with pytest.raises(ValueError, match="breadth must be a positive integer"):
        StepDataOrdering(2, True)  # type: ignore[arg-type]


def test_label_names_the_walk_and_its_flags() -> None:
    assert StepDataOrdering(2, 4).label == "2x4rp"
    assert StepDataOrdering(2, 4, False, False).label == "2x4"
    assert StepDataOrdering.depth_first(8).label == "8x1rp"


def test_passes_walk_forward_and_backward_as_asked() -> None:
    ordering = StepDataOrdering(2, 4)
    assert ordering.positions(1) == range(4, 8)
    assert ordering.backward_positions(1) == (7, 6, 5, 4)
    assert StepDataOrdering(2, 4, reverse_breadth=False).backward_positions(1) == (
        4,
        5,
        6,
        7,
    )
    with pytest.raises(IndexError):
        ordering.positions(2)


def test_the_first_backward_the_walk_reaches_creates_the_gradient() -> None:
    reversed_paired = StepDataOrdering(2, 4)
    # every ordinary stage: the pass's last microbatch goes first in backward
    assert [reversed_paired.creates(p, 0, stage_count=3) for p in range(8)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    # the paired last stage walks in forward order, so its first microbatch creates
    assert [reversed_paired.creates(p, 2, stage_count=3) for p in range(8)] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    plain = StepDataOrdering(2, 4, reverse_breadth=False, pair_loss=False)
    assert all(plain.creates(0, stage, stage_count=3) for stage in range(3))
    assert not any(plain.creates(p, 1, stage_count=3) for p in range(1, 8))
    depth_first = StepDataOrdering.depth_first(4)
    assert [depth_first.creates(p, 1, stage_count=3) for p in range(4)] == [
        True,
        False,
        False,
        False,
    ]


def test_round_trips_through_its_dict_and_tolerates_old_records() -> None:
    ordering = StepDataOrdering(2, 4, False, True)
    assert StepDataOrdering.from_dict(ordering.to_dict(), "x") == ordering
    assert StepDataOrdering.from_dict({"depth": 1, "breadth": 2}, "x") == (
        StepDataOrdering(1, 2)
    )
    with pytest.raises(ValueError, match="x: "):
        StepDataOrdering.from_dict({"depth": 1}, "x")


def test_labels_round_trip() -> None:
    for ordering in (
        StepDataOrdering(2, 4),
        StepDataOrdering(2, 4, False, False),
        StepDataOrdering(1, 8, True, False),
    ):
        assert StepDataOrdering.from_label(ordering.label) == ordering
    with pytest.raises(ValueError, match="not <depth>x<breadth>"):
        StepDataOrdering.from_label("2by4")
