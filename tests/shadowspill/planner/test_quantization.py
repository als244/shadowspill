"""Planning inputs coarsen only once they reach one quantum."""

from __future__ import annotations

from shadowspill.planner.quantization import (
    GIBIBYTE,
    GIGABYTE_PER_SECOND,
    MICROSECOND_NS,
    floored,
    nearest,
)


def test_small_values_stay_exact() -> None:
    assert floored(1024, GIBIBYTE) == 1024
    assert nearest(100, GIGABYTE_PER_SECOND) == 100
    assert nearest(10, MICROSECOND_NS) == 10


def test_budgets_round_down_to_whole_gibibytes() -> None:
    assert floored(111 * GIBIBYTE + GIBIBYTE - 1, GIBIBYTE) == 111 * GIBIBYTE
    assert floored(112 * GIBIBYTE, GIBIBYTE) == 112 * GIBIBYTE


def test_rates_and_latencies_round_to_the_nearest_quantum() -> None:
    assert nearest(25_400_000_000, GIGABYTE_PER_SECOND) == 25_000_000_000
    assert nearest(25_600_000_000, GIGABYTE_PER_SECOND) == 26_000_000_000
    assert nearest(8_600, MICROSECOND_NS) == 9_000
    assert nearest(8_499, MICROSECOND_NS) == 8_000
