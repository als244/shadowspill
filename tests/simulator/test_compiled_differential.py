from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import given
from hypothesis import strategies as st

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectRole,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.simulator import SimulationConfig, SimulationInfeasibleError
from shadowspill.simulator._compiled import simulate_compiled
from shadowspill.simulator._python import simulate_python
from tests.ir._examples import (
    SAVE_SELECTION,
    representative_program,
    representative_schedule,
)

from ._examples import (
    calibrated_config,
    concurrent_lane_program,
    initial_only_schedule,
    ordered_action_program,
    ordered_action_schedule,
    overlap_program,
    overlap_schedule,
)

pytestmark = pytest.mark.skipif(
    "SHADOWSPILL_SIMULATOR_LIBRARY" not in os.environ,
    reason="compiled simulator library was not supplied",
)


@pytest.mark.parametrize(
    ("program", "schedule", "selections", "capacity"),
    [
        (
            representative_program(),
            representative_schedule(),
            SAVE_SELECTION,
            600,
        ),
        (overlap_program(), overlap_schedule(), (), 512),
        (concurrent_lane_program(), initial_only_schedule(), (), 1024),
        (ordered_action_program(), ordered_action_schedule(), (), 1024),
    ],
)
def test_compiled_and_python_results_are_identical(
    program: Program,
    schedule: MemorySchedule,
    selections: tuple,
    capacity: int,
) -> None:
    config = calibrated_config(device_capacity_bytes=capacity)

    expected = simulate_python(
        program,
        schedule,
        selections=selections,
        config=config,
    )
    actual = simulate_compiled(
        program,
        schedule,
        selections=selections,
        config=config,
    )

    assert actual == expected


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_compiled_replay_is_deterministic_across_worker_counts(workers: int) -> None:
    program = overlap_program()
    schedule = overlap_schedule()
    config = calibrated_config(device_capacity_bytes=512)
    expected = simulate_python(program, schedule, config=config)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(
            executor.map(
                lambda _: simulate_compiled(program, schedule, config=config),
                range(32),
            )
        )

    assert all(result == expected for result in results)


@pytest.mark.parametrize("capacity", [319, 527])
def test_compiled_and_python_capacity_diagnostics_agree(capacity: int) -> None:
    config = calibrated_config(device_capacity_bytes=capacity)

    with pytest.raises(SimulationInfeasibleError) as python_caught:
        simulate_python(
            representative_program(),
            representative_schedule(),
            selections=SAVE_SELECTION,
            config=config,
        )
    with pytest.raises(SimulationInfeasibleError) as compiled_caught:
        simulate_compiled(
            representative_program(),
            representative_schedule(),
            selections=SAVE_SELECTION,
            config=config,
        )

    python_error = python_caught.value
    compiled_error = compiled_caught.value
    assert str(compiled_error) == str(python_error)
    assert compiled_error.kind == python_error.kind
    assert compiled_error.task_id == python_error.task_id
    assert compiled_error.location == python_error.location
    assert compiled_error.capacity_bytes == python_error.capacity_bytes
    assert compiled_error.used_bytes == python_error.used_bytes
    assert compiled_error.requested_bytes == python_error.requested_bytes


def test_large_integer_transfer_runtime_does_not_overflow() -> None:
    size = (1 << 63) + 17
    bandwidth = (1 << 63) + 101
    resource = ResourceSpec("cuda_0", ResourceKind.CONTROL)
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(AliasGroupSpec("storage", "cuda_0", size),),
        objects=(ObjectSpec("object", "storage", 0, size),),
        profiles=(TaskProfile("trigger_profile", 0, 0, "trigger_abi"),),
        tasks=(
            TaskSpec(
                "trigger",
                resource,
                "trigger_profile",
                requires_entrypoint=False,
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=(ResidencySpec("storage", MemoryLocation.HOST),),
        actions=(
            MemoryAction(
                "trigger",
                "storage",
                MemoryActionKind.PREFETCH,
            ),
        ),
        final_residency=(ResidencySpec("storage", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=size,
        host_capacity_bytes=size,
        fetch_bandwidth_bytes_per_second=bandwidth,
        evict_bandwidth_bytes_per_second=bandwidth,
    )

    expected = simulate_python(program, schedule, config=config)
    actual = simulate_compiled(program, schedule, config=config)

    assert actual == expected
    assert actual.makespan_ns == 1_000_000_000


@given(
    sizes=st.lists(st.integers(min_value=0, max_value=4096), min_size=1, max_size=12),
    runtimes=st.lists(
        st.integers(min_value=0, max_value=1_000_000), min_size=1, max_size=12
    ),
    workspaces=st.lists(
        st.integers(min_value=0, max_value=1024), min_size=1, max_size=12
    ),
)
def test_random_linear_programs_match(
    sizes: list[int],
    runtimes: list[int],
    workspaces: list[int],
) -> None:
    count = min(len(sizes), len(runtimes), len(workspaces))
    sizes = sizes[:count]
    runtimes = runtimes[:count]
    workspaces = workspaces[:count]
    resource = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    aliases = tuple(
        AliasGroupSpec(f"storage_{index}", "cuda_0", size)
        for index, size in enumerate(sizes)
    )
    objects = tuple(
        ObjectSpec(
            f"object_{index}",
            f"storage_{index}",
            0,
            size,
            ObjectRole.OUTPUT,
        )
        for index, size in enumerate(sizes)
    )
    profiles = tuple(
        TaskProfile(
            f"profile_{index}",
            runtimes[index],
            workspaces[index],
            f"abi_{index}",
        )
        for index in range(count)
    )
    tasks = tuple(
        TaskSpec(
            f"task_{index}",
            resource,
            f"profile_{index}",
            dependencies=(() if index == 0 else (f"task_{index - 1}",)),
            outputs=(f"object_{index}",),
        )
        for index in range(count)
    )
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=aliases,
        objects=objects,
        profiles=profiles,
        tasks=tasks,
    )
    schedule = MemorySchedule(
        initial_residency=(),
        actions=(),
        final_residency=(ResidencySpec(f"storage_{count - 1}", MemoryLocation.DEVICE),),
    )
    capacity = sum(sizes) + max(workspaces)
    config = calibrated_config(device_capacity_bytes=capacity)

    expected = simulate_python(program, schedule, config=config)
    actual = simulate_compiled(program, schedule, config=config)

    assert actual == expected
