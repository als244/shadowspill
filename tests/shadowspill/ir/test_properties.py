from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
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
    index_program,
)


@given(
    sizes=st.lists(
        st.integers(min_value=0, max_value=1 << 30), min_size=1, max_size=20
    ),
    runtimes=st.lists(
        st.integers(min_value=0, max_value=1 << 50), min_size=1, max_size=20
    ),
)
def test_linear_programs_round_trip_with_stable_plan_indexs(
    sizes: list[int],
    runtimes: list[int],
) -> None:
    count = min(len(sizes), len(runtimes))
    sizes = sizes[:count]
    runtimes = runtimes[:count]
    aliases = tuple(
        AliasGroupSpec(f"storage_{index}", "cuda_0", size)
        for index, size in enumerate(sizes)
    )
    objects = tuple(
        ObjectSpec(
            f"object_{index}",
            alias.alias_group_id,
            0,
            alias.size_bytes,
            ObjectRole.INPUT if index == 0 else ObjectRole.ACTIVATION,
        )
        for index, alias in enumerate(aliases)
    )
    profiles = tuple(
        TaskProfile(f"profile_{index}", runtime, index, f"abi_{index}")
        for index, runtime in enumerate(runtimes)
    )
    resource = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    tasks = tuple(
        TaskSpec(
            f"task_{index}",
            resource,
            profiles[index].profile_id,
            dependencies=(() if index == 0 else (f"task_{index - 1}",)),
            inputs=((objects[0].object_id,) if index == 0 else ()),
            outputs=(() if index == 0 else (objects[index].object_id,)),
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
        initial_residency=(ResidencySpec("storage_0", MemoryLocation.DEVICE),),
        actions=(),
    )

    restored = Program.from_json(program.to_json())

    assert restored == program
    assert index_program(restored) == index_program(program)
    schedule.validate(program)
