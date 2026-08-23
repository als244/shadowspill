from __future__ import annotations

from dataclasses import replace

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    ObjectSpec,
    Program,
    RecomputationGroup,
    RecomputationOption,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner.recomputation import resolutions

DEVICE = DeviceSpec("cuda_0", "process_0", "cuda", 0)
COMPUTE = ResourceSpec("cuda_0", ResourceKind.COMPUTE)


def _binary_program(group_count: int) -> Program:
    aliases = tuple(
        AliasGroupSpec(f"saved_{index}", "cuda_0", 10) for index in range(group_count)
    )
    objects = tuple(
        ObjectSpec(f"saved_object_{index}", f"saved_{index}", 0, 10)
        for index in range(group_count)
    )
    profiles = (
        TaskProfile("save_profile", 10, 0, "save_abi"),
        TaskProfile("recompute_profile", 20, 0, "recompute_abi"),
    )
    tasks: list[TaskSpec] = []
    groups: list[RecomputationGroup] = []
    for index in range(group_count):
        save_task = f"save_task_{index}"
        recompute_task = f"recompute_task_{index}"
        tasks.extend(
            (
                TaskSpec(save_task, COMPUTE, "save_profile"),
                TaskSpec(recompute_task, COMPUTE, "recompute_profile"),
            )
        )
        groups.append(
            RecomputationGroup(
                f"choice_{index}",
                (
                    RecomputationOption(
                        "save",
                        (save_task,),
                        (f"saved_{index}",),
                    ),
                    RecomputationOption("recompute", (recompute_task,)),
                ),
            )
        )
    return Program(
        devices=(DEVICE,),
        alias_groups=aliases,
        objects=objects,
        profiles=profiles,
        tasks=tuple(tasks),
        recomputation_groups=tuple(groups),
    )


def _option_ids(program: Program) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(selection.option_id for selection in selections)
        for selections in resolutions(program)
    )


def test_a_small_inventory_is_resolved_exhaustively() -> None:
    options = _option_ids(_binary_program(6))

    assert len(options) == 64
    assert options[0] == ("save",) * 6
    assert options[-1] == ("recompute",) * 6


def test_a_large_binary_inventory_uses_even_group_quarters() -> None:
    options = _option_ids(_binary_program(8))

    assert tuple(item.count("recompute") for item in options) == (0, 2, 4, 6, 8)
    assert options[0] == ("save",) * 8
    assert options[1] == (
        "save",
        "save",
        "recompute",
        "save",
        "save",
        "save",
        "recompute",
        "save",
    )
    assert options[2] == (
        "save",
        "recompute",
        "save",
        "recompute",
        "save",
        "recompute",
        "save",
        "recompute",
    )
    assert options[-1] == ("recompute",) * 8


def test_a_large_inventory_is_bounded() -> None:
    options = _option_ids(_binary_program(64))

    assert len(options) == 5
    assert tuple(item.count("recompute") for item in options) == (
        0,
        16,
        32,
        48,
        64,
    )


def test_terminal_forward_group_is_always_saved() -> None:
    program = _binary_program(8)
    tasks: list[TaskSpec] = []
    for group_index in range(8):
        dependencies = (
            ()
            if group_index == 0
            else (
                f"save_task_{group_index - 1}",
                f"recompute_task_{group_index - 1}",
            )
        )
        tasks.extend(
            replace(task, phase="forward", dependencies=dependencies)
            for task in program.tasks[2 * group_index : 2 * group_index + 2]
        )
    linear = replace(program, tasks=tuple(tasks))

    options = _option_ids(linear)

    assert len(options) == 5
    assert tuple(item.count("recompute") for item in options) == (0, 2, 4, 5, 7)
    assert all(item[-1] == "save" for item in options)
