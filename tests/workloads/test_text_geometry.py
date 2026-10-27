"""Text-specific microbatch enumeration stays outside the planner."""

import pytest

from workloads.recipes.text.geometry import search_geometries


def test_geometries_cover_every_divisor_largest_microbatch_first() -> None:
    admitted, skipped = search_geometries(12, sequence_length=1)
    assert admitted == ((12, 1), (6, 2), (4, 3), (3, 4), (2, 6), (1, 12))
    assert skipped == ()


def test_bounds_skip_with_reasons_rather_than_silently() -> None:
    admitted, skipped = search_geometries(
        12,
        sequence_length=1024,
        min_tokens_per_microbatch=3 * 1024,
        max_tokens_per_microbatch=6 * 1024,
    )
    assert admitted == ((6, 2), (4, 3), (3, 4))
    assert [(item[0], item[1]) for item in skipped] == [(12, 1), (2, 6), (1, 12)]
    assert all(item[2] for item in skipped)


def test_a_non_positive_total_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        search_geometries(0, sequence_length=1)
