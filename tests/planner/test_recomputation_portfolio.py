from __future__ import annotations

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
from shadowspill.planner._recomputation import build_recomputation_portfolio

DEVICE = DeviceSpec("cuda_0", "process_0", "cuda", 0)
COMPUTE = ResourceSpec("cuda_0", ResourceKind.COMPUTE)


def _binary_program(group_count: int) -> Program:
    aliases = tuple(
        AliasGroupSpec(f"saved_{index}", "cuda_0", 10)
        for index in range(group_count)
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
        for selections in build_recomputation_portfolio(program)
    )


def test_small_recomputation_portfolio_remains_exhaustive() -> None:
    options = _option_ids(_binary_program(6))

    assert len(options) == 64
    assert options[0] == ("save",) * 6
    assert options[-1] == ("recompute",) * 6


def test_large_binary_portfolio_uses_four_deterministic_seeds() -> None:
    options = _option_ids(_binary_program(7))

    assert options == (
        ("save",) * 7,
        ("recompute",) * 7,
        (
            "recompute",
            "save",
            "recompute",
            "save",
            "recompute",
            "save",
            "recompute",
        ),
        (
            "save",
            "recompute",
            "save",
            "recompute",
            "save",
            "recompute",
            "save",
        ),
    )
