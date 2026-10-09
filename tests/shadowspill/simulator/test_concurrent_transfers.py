"""Exact lane-overlap timing, including fractional bytes and ABI limits."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference.python.simulator import simulate_python
from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.simulator import SimulationConfig, simulate
from shadowspill.simulator.model import DeviceSimulationConfig

G = 1_000_000_000
MAX = (1 << 64) - 1


def replay(copies, *, rates=(2 * G, G, 2 * G, G), latencies=(0, 0)):
    """(direction, size, ready time, device) copies; no compute contention."""
    device_ids = sorted({device for _, _, _, device in copies})
    arrivals = sorted({ready for _, _, ready, _ in copies})
    tasks = []
    profiles = []
    previous = 0
    for i, ready in enumerate(arrivals):
        profiles.append(TaskProfile(f"p{i}", ready - previous, 0, f"abi{i}"))
        tasks.append(
            TaskSpec(
                f"t{i}",
                ResourceSpec(device_ids[0], ResourceKind.CONTROL),
                f"p{i}",
                dependencies=() if i == 0 else (f"t{i - 1}",),
            )
        )
        previous = ready
    aliases = tuple(
        AliasGroupSpec(f"a{i}", device, size)
        for i, (_, size, _, device) in enumerate(copies)
    )
    program = ShadowSpillProgram(
        devices=tuple(
            DeviceSpec(device, "process_0", "cuda", i)
            for i, device in enumerate(device_ids)
        ),
        alias_groups=aliases,
        objects=tuple(
            ObjectSpec(f"o{i}", a.alias_group_id, 0, a.size_bytes)
            for i, a in enumerate(aliases)
        ),
        tasks=tuple(tasks),
        profiles=tuple(profiles),
    )
    schedule = MemorySchedule(
        initial_residency=tuple(
            ResidencySpec(
                f"a{i}",
                MemoryLocation.SPILL if direction == "fetch" else MemoryLocation.DEVICE,
            )
            for i, (direction, _, _, _) in enumerate(copies)
        ),
        actions=tuple(
            MemoryAction(
                f"t{arrivals.index(ready)}",
                f"a{i}",
                MemoryActionKind.FETCH
                if direction == "fetch"
                else MemoryActionKind.EVICT,
            )
            for i, (direction, _, ready, _) in sorted(
                enumerate(copies), key=lambda pair: (pair[1][2], pair[0])
            )
        ),
        final_residency=tuple(
            ResidencySpec(
                f"a{i}",
                MemoryLocation.DEVICE if direction == "fetch" else MemoryLocation.SPILL,
            )
            for i, (direction, _, _, _) in enumerate(copies)
        ),
    )
    config = SimulationConfig(
        tuple(
            DeviceSimulationConfig(device, MAX, *rates, *latencies)
            for device in device_ids
        ),
        MAX,
    )
    actual = simulate(program, schedule, config=config)
    expected = simulate_python(program, schedule, config=config)
    assert actual == expected
    return {t.alias_group_id: (t.start_ns, t.end_ns) for t in actual.transfer_intervals}


@pytest.mark.parametrize(
    ("copies", "expected"),
    [
        ([("fetch", 100, 0, "d")], {"a0": (0, 50)}),
        ([("evict", 100, 0, "d")], {"a0": (0, 50)}),
        (
            [("fetch", 100, 0, "d"), ("evict", 100, 0, "d")],
            {"a0": (0, 100), "a1": (0, 100)},
        ),
        (
            [("fetch", 100, 0, "d"), ("evict", 100, 25, "d")],
            {"a0": (0, 75), "a1": (25, 100)},
        ),
        (
            [("fetch", 200, 0, "d"), ("evict", 100, 0, "d")],
            {"a0": (0, 150), "a1": (0, 100)},
        ),
        (
            [("fetch", 100, 0, "d"), ("evict", 100, 60, "d")],
            {"a0": (0, 50), "a1": (60, 110)},
        ),
        (
            [("fetch", 200, 0, "d"), ("evict", 40, 0, "d"), ("evict", 60, 0, "d")],
            {"a0": (0, 150), "a1": (0, 40), "a2": (40, 100)},
        ),
        (
            [("fetch", 100, 0, "d0"), ("evict", 100, 0, "d1")],
            {"a0": (0, 50), "a1": (0, 50)},
        ),
    ],
)
@pytest.mark.parametrize("latencies", [(0, 0), (200_000, 300_000), (MAX, MAX)])
def test_solo_full_partial_queued_and_independent_lanes(copies, expected, latencies):
    # Metadata, even at the ABI limit, must not delay either lane or change
    # the overlap intervals that choose between solo and concurrent rates.
    assert replay(copies, latencies=latencies) == expected


def test_sub_byte_progress_is_not_discarded_at_overlap_boundaries():
    # Fetch moves 1/3 byte per ns, then 1/7 with the evict. Preserving the
    # first fraction is observable even for these tiny synthetic transfers.
    replay(
        [("fetch", 10, 0, "d"), ("evict", 3, 1, "d"), ("evict", 2, 14, "d")],
        rates=(333_333_333, 142_857_143, 777_777_777, 222_222_222),
    )


@pytest.mark.parametrize(
    ("size", "rates"),
    [
        (MAX // 4, (MAX, MAX // 3, MAX, MAX // 5)),
        (MAX // 4, (G + 3, G - 9, G + 7, G - 21)),
        (1, (1, 1, 2, 1)),
    ],
)
def test_integer_limits_without_overflow(size, rates):
    replay([("fetch", size, 0, "d"), ("evict", size, 17, "d")], rates=rates)


@given(
    copies=st.lists(
        st.tuples(
            st.sampled_from(["fetch", "evict"]),
            st.integers(1, 10000),
            st.integers(0, 1000),
            st.sampled_from(["a", "b"]),
        ),
        min_size=1,
        max_size=12,
    ),
    rates=st.tuples(*(st.integers(1, 100 * G) for _ in range(4))),
    latencies=st.tuples(st.integers(0, 100), st.integers(0, 100)),
)
@settings(max_examples=200, deadline=None)
def test_arbitrary_lane_events_match_independent_integer_oracle(
    copies, rates, latencies
):
    replay(copies, rates=rates, latencies=latencies)


@pytest.mark.parametrize(
    "field",
    [
        "fetch_solo_bandwidth_bytes_per_second",
        "fetch_concurrent_bandwidth_bytes_per_second",
        "evict_solo_bandwidth_bytes_per_second",
        "evict_concurrent_bandwidth_bytes_per_second",
    ],
)
def test_every_rate_must_be_a_positive_integer(field):
    config = DeviceSimulationConfig("d", 1024, 2 * G, G, 2 * G, G)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match=field):
            replace(config, **{field: value})
