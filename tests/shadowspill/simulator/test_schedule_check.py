"""The library's schedule check against the readable rules it implements.

`MemorySchedule.validate` states the rules; `check_schedule` applies them in
the library on a bound schedule. For random programs and schedules -- most of
them refused somewhere -- both must reach the same verdict and, when they
refuse, the same first refusal: the same path and the same message.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

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
from shadowspill.ir.program import MutationSpec, SharedResidencyPolicy
from shadowspill.ir.validation import ValidationError
from shadowspill.libraries import shadowspill_library_path
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.indexing import check_schedule, index_simulation_template

pytestmark = pytest.mark.skipif(
    shadowspill_library_path() is None,
    reason="the simulator library is not installed",
)

_KINDS = tuple(MemoryActionKind)
_LOCATIONS = tuple(MemoryLocation)


def _program(rng: random.Random) -> ShadowSpillProgram:
    """Alias groups of every sort -- plain, retained, empty and shared -- one
    object each, and a chain of tasks reading and writing them."""

    groups, objects = [], []
    for index in range(rng.randint(3, 9)):
        sort = rng.choice(("plain", "retained", "empty", "shared"))
        size = 0 if sort == "empty" else 64
        groups.append(
            AliasGroupSpec(
                f"group_{index}",
                "device_0",
                size,
                retain_spill_copy=sort in ("retained", "shared"),
                shared_residency=(
                    SharedResidencyPolicy.SHARED_READ_ONLY if sort == "shared" else None
                ),
            )
        )
        objects.append(ObjectSpec(f"object_{index}", f"group_{index}", 0, size))
    writable = [
        item.object_id
        for item, group in zip(objects, groups, strict=True)
        if group.shared_residency is None
    ]
    rng.shuffle(writable)
    produced: dict[str, str] = {}
    tasks = []
    for index in range(rng.randint(2, 8)):
        task_id = f"task_{index}"
        output = writable.pop() if writable and rng.random() < 0.8 else None
        candidates = [item.object_id for item in objects if item.object_id != output]
        inputs = tuple(
            rng.sample(candidates, k=min(len(candidates), rng.randint(0, 2)))
        )
        dependencies = {f"task_{index - 1}"} if index else set()
        dependencies.update(produced[item] for item in inputs if item in produced)
        mutable = [
            item
            for item in inputs
            if groups[int(item.split("_")[1])].shared_residency is None
        ]
        mutations = (
            (MutationSpec(rng.choice(mutable)),)
            if mutable and rng.random() < 0.3
            else ()
        )
        tasks.append(
            TaskSpec(
                task_id,
                ResourceSpec("device_0", ResourceKind.COMPUTE),
                "profile",
                dependencies=tuple(sorted(dependencies)),
                inputs=inputs,
                outputs=(output,) if output else (),
                mutations=mutations,
            )
        )
        if output:
            produced[output] = task_id
    return ShadowSpillProgram(
        devices=(DeviceSpec("device_0", "process_0", "device", 0),),
        alias_groups=tuple(groups),
        objects=tuple(objects),
        profiles=(TaskProfile("profile", 10, 0, "contract"),),
        tasks=tuple(tasks),
    )


def _schedule(rng: random.Random, program: ShadowSpillProgram) -> MemorySchedule:
    """A schedule that is plausible more often than not: movable groups,
    actions in trigger order, residency declared once per group -- with a
    small chance of each thing the rules refuse."""

    groups = program.alias_groups
    movable = [
        g.alias_group_id for g in groups if g.shared_residency is None and g.size_bytes
    ]
    anything = [g.alias_group_id for g in groups]
    tasks = [task.task_id for task in program.tasks]

    def pick(pool: list[str]) -> str:
        return rng.choice(anything if not pool or rng.random() < 0.08 else pool)

    initial = {pick(movable): rng.choice(_LOCATIONS) for _ in range(rng.randint(0, 4))}
    triggers = sorted(rng.randrange(len(tasks)) for _ in range(rng.randint(0, 8)))
    if triggers and rng.random() < 0.1:
        rng.shuffle(triggers)
    actions = tuple(
        MemoryAction(tasks[trigger], pick(movable), rng.choice(_KINDS))
        for trigger in triggers
    )
    final = {pick(movable): rng.choice(_LOCATIONS) for _ in range(rng.randint(0, 3))}
    return MemorySchedule(
        initial_residency=tuple(ResidencySpec(a, loc) for a, loc in initial.items()),
        actions=actions,
        final_residency=tuple(ResidencySpec(a, loc) for a, loc in final.items()),
    )


def _verdict(call, *arguments) -> tuple[str, str] | None:
    try:
        call(*arguments)
    except ValidationError as error:
        return error.path, error.message
    return None


def test_the_library_check_refuses_exactly_what_the_rules_refuse() -> None:
    rng = random.Random(20261011)
    config = SimulationConfig.single_device(
        "device_0",
        device_capacity_bytes=1 << 20,
        spill_capacity_bytes=1 << 20,
        fetch_solo_bandwidth_bytes_per_second=1_000_000,
        fetch_concurrent_bandwidth_bytes_per_second=1_000_000,
        evict_solo_bandwidth_bytes_per_second=1_000_000,
        evict_concurrent_bandwidth_bytes_per_second=1_000_000,
    )
    seen: Counter[str] = Counter()
    for _ in range(3000):
        program = _program(rng)
        template = index_simulation_template(program, (), config)
        for _ in range(5):
            schedule = _schedule(rng, program)
            expected = _verdict(schedule.validate, program)
            actual = _verdict(check_schedule, template, schedule)
            assert actual == expected, (program, schedule)
            seen["valid" if expected is None else expected[1].split("'")[0]] += 1
    # Every rule is exercised, and so is a schedule that passes them all.
    assert seen["valid"] > 0
    for message in (
        "shared residency is owned by the runtime, not the schedule",
        "shared alias groups cannot have plan-owned memory actions",
        "zero-size alias groups cannot have physical memory actions",
        "actions must be ordered by trigger task",
        "alias group ",
        "fetch requires current host residency",
        "fetch requires absent device residency",
        "evict requires device residency",
        "release requires device residency",
        "write-back requires device residency",
        "release drops the only current copy of a value still needed",
        "required current device residency was not reached",
        "required current host residency was not reached",
    ):
        assert any(key.startswith(message) for key in seen), (message, seen)
