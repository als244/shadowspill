"""Planning inputs coarsen by magnitude, and stay exact below one quantum."""

from __future__ import annotations

from shadowspill.planner.quantization import (
    GIBIBYTE,
    GIGABYTE_PER_SECOND,
    MICROSECOND_NS,
    floored,
    nearest,
    quantized_bandwidth,
    quantized_latency,
)


def test_small_values_stay_exact() -> None:
    assert floored(1024, GIBIBYTE) == 1024
    assert nearest(100, GIGABYTE_PER_SECOND) == 100
    assert nearest(10, MICROSECOND_NS) == 10


def test_budgets_round_down_to_whole_gibibytes() -> None:
    assert floored(111 * GIBIBYTE + GIBIBYTE - 1, GIBIBYTE) == 111 * GIBIBYTE
    assert floored(112 * GIBIBYTE, GIBIBYTE) == 112 * GIBIBYTE


def test_a_fast_lane_rounds_to_half_a_gigabyte_per_second() -> None:
    half = GIGABYTE_PER_SECOND // 2
    assert quantized_bandwidth(25_400_000_000) == 25_500_000_000
    assert quantized_bandwidth(25_600_000_000) == 25_500_000_000
    assert quantized_bandwidth(26_400_000_000) == 26_500_000_000
    assert quantized_bandwidth(half) == half


def test_a_slow_lane_rounds_to_a_tenth_of_a_gigabyte_per_second() -> None:
    tenth = GIGABYTE_PER_SECOND // 10
    assert quantized_bandwidth(260_000_000) == 3 * tenth
    assert quantized_bandwidth(440_000_000) == 4 * tenth
    # Below one tenth there is no quantum to reach, so the value stays exact.
    assert quantized_bandwidth(70_000_000) == 70_000_000


def test_latency_quantum_grows_with_the_latency() -> None:
    us = MICROSECOND_NS
    # below five microseconds: to the microsecond
    assert quantized_latency(1_400) == 1 * us
    assert quantized_latency(2_600) == 3 * us
    # five microseconds and above: to five
    assert quantized_latency(5 * us) == 5 * us
    assert quantized_latency(7 * us) == 5 * us
    assert quantized_latency(12 * us) == 10 * us
    # above one hundred: to fifty
    assert quantized_latency(140 * us) == 150 * us
    # above five hundred: to one hundred
    assert quantized_latency(560 * us) == 600 * us
    # above a millisecond: to two hundred and fifty
    assert quantized_latency(1_130 * us) == 1_250 * us
    # above ten milliseconds: to the millisecond
    assert quantized_latency(12_600 * us) == 13_000 * us


def test_each_latency_band_keeps_its_own_boundary() -> None:
    us = MICROSECOND_NS
    # A boundary belongs to the finer band: the coarser one applies *above* it.
    assert quantized_latency(100 * us) == 100 * us
    assert quantized_latency(101 * us) == 100 * us
    assert quantized_latency(500 * us) == 500 * us
    assert quantized_latency(1_000 * us) == 1_000 * us
    assert quantized_latency(10_000 * us) == 10_000 * us
