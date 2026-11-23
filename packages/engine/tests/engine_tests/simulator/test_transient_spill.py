"""Which spill leases a schedule holds only while it runs, and which it keeps."""

from __future__ import annotations

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
DEVICE, SPILL = MemoryLocation.DEVICE, MemoryLocation.SPILL
FETCH, EVICT = MemoryActionKind.FETCH, MemoryActionKind.EVICT


def run(aliases, initial, actions, final, *, reads=()):
    """Simulate three tasks in a row; ``consumer`` reads ``reads``."""

    program = ShadowSpillProgram(
        (DeviceSpec("gpu"),),
        tuple(
            AliasGroupSpec(name, "gpu", size, **options)
            for name, size, options in aliases
        ),
        tuple(ObjectSpec(name, name, 0, size) for name, size, _ in aliases),
        (TaskProfile("empty", 0, 0, "v1"), TaskProfile("use", 10, 0, "v1")),
        (
            TaskSpec("entry", ResourceSpec("gpu", ResourceKind.CONTROL), "empty"),
            TaskSpec(
                "middle",
                ResourceSpec("gpu", ResourceKind.COMPUTE),
                "use",
                dependencies=("entry",),
            ),
            TaskSpec(
                "consumer",
                ResourceSpec("gpu", ResourceKind.COMPUTE),
                "use",
                dependencies=("middle",),
                inputs=tuple(reads),
            ),
        ),
    )
    schedule = MemorySchedule(
        tuple(ResidencySpec(name, where) for name, where in initial),
        tuple(MemoryAction(task, name, kind) for task, name, kind in actions),
        tuple(ResidencySpec(name, where) for name, where in final),
    )
    config = SimulationConfig(
        (DeviceSimulationConfig("gpu", 10_000, G, G, G, G),), 10_000
    )
    return simulate(program, schedule, config=config)


def test_a_lease_the_residency_places_in_spill_is_persistent():
    result = run(
        [("state", 400, {})],
        [("state", SPILL)],
        [("entry", "state", FETCH)],
        [("state", DEVICE)],
        reads=("state",),
    )
    assert result.spill_peak_bytes == 400
    assert result.spill_transient_peak_bytes == 0
    assert result.spill_transient_end_bytes == 0


def test_a_value_evicted_and_fetched_back_is_transient():
    result = run(
        [("scratch", 300, {})],
        [("scratch", DEVICE)],
        [("entry", "scratch", EVICT), ("middle", "scratch", FETCH)],
        [("scratch", DEVICE)],
        reads=("scratch",),
    )
    assert result.spill_peak_bytes == 300
    assert result.spill_transient_peak_bytes == 300
    assert result.spill_transient_end_bytes == 0


def test_transient_spill_a_schedule_leaves_behind_is_reported():
    result = run(
        [("left", 200, {})],
        [("left", DEVICE)],
        [("entry", "left", EVICT)],
        [],
    )
    assert result.spill_transient_peak_bytes == 200
    assert result.spill_transient_end_bytes == 200


def test_a_retained_spill_copy_is_persistent():
    result = run(
        [("copy", 100, {"retain_spill_copy": True}), ("scratch", 300, {})],
        [("copy", DEVICE), ("scratch", DEVICE)],
        [("entry", "scratch", EVICT), ("middle", "scratch", FETCH)],
        [("copy", DEVICE), ("scratch", DEVICE)],
        reads=("copy", "scratch"),
    )
    assert result.spill_peak_bytes == 400
    assert result.spill_transient_peak_bytes == 300
    assert result.spill_transient_end_bytes == 0


def test_a_lease_filled_only_for_the_final_residency_is_held_from_the_start():
    # The scratch value is back on the device before "gradient" is evicted;
    # the gradient's lease still counts beside it.
    result = run(
        [("gradient", 500, {}), ("scratch", 300, {})],
        [("gradient", DEVICE), ("scratch", DEVICE)],
        [
            ("entry", "scratch", EVICT),
            ("middle", "scratch", FETCH),
            ("consumer", "gradient", EVICT),
        ],
        [("gradient", SPILL), ("scratch", DEVICE)],
        reads=("scratch",),
    )
    assert result.spill_peak_bytes == 800
    assert result.spill_transient_peak_bytes == 300
    assert result.spill_transient_end_bytes == 0


def test_a_fetched_persistent_lease_stays_held():
    # "state" leaves spill before the scratch value arrives; its lease
    # stays reserved, so the two add up.
    result = run(
        [("state", 400, {}), ("scratch", 300, {})],
        [("state", SPILL), ("scratch", DEVICE)],
        [
            ("entry", "state", FETCH),
            ("middle", "scratch", EVICT),
            ("consumer", "scratch", FETCH),
        ],
        [("state", DEVICE), ("scratch", DEVICE)],
        reads=("state",),
    )
    assert result.spill_peak_bytes == 700
    assert result.spill_transient_peak_bytes == 300
