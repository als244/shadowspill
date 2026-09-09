from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskAlternativeGroup,
    TaskAlternativeOption,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import GenericPlanningOptions, pressurefit
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions
from shadowspill.planner.search.toolkit.resolution import (
    DEFAULT_RESOLUTION_OPTIONS,
    CostedAlternatives,
    resolutions,
    validate_resolution_options,
)
from shadowspill.simulator import SimulationConfig

DEVICE = DeviceSpec("cuda_0", "process_0", "cuda", 0)
COMPUTE = ResourceSpec("cuda_0", ResourceKind.COMPUTE)


def _binary_program(group_count: int) -> ShadowSpillProgram:
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
    groups: list[TaskAlternativeGroup] = []
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
            TaskAlternativeGroup(
                f"choice_{index}",
                (
                    TaskAlternativeOption(
                        "save",
                        (save_task,),
                        (f"saved_{index}",),
                    ),
                    TaskAlternativeOption("recompute", (recompute_task,)),
                ),
            )
        )
    return ShadowSpillProgram(
        devices=(DEVICE,),
        alias_groups=aliases,
        objects=objects,
        profiles=profiles,
        tasks=tuple(tasks),
        task_alternative_groups=tuple(groups),
    )


def _option_ids(
    program: ShadowSpillProgram,
    shares: tuple[Fraction | int | str, ...] = DEFAULT_RESOLUTION_OPTIONS,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(selection.option_id for selection in selections)
        for selections in resolutions(program, shares)
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
    assert tuple(item.count("recompute") for item in options) == tuple(range(0, 65, 16))


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

    # seven flexible groups: the quarters round half up to 0, 2, 4, 5 and 7
    assert len(options) == 5
    assert tuple(item.count("recompute") for item in options) == (0, 2, 4, 5, 7)
    assert all(item[-1] == "save" for item in options)


def test_a_group_whose_options_keep_the_same_bytes_is_not_searched() -> None:
    """An alternative trades runtime for retained bytes. One that retains the
    same either way trades nothing, so it is one plan spelled twice and the
    search should not carry it as a dimension."""

    program = _binary_program(4)
    # strip the first group's saved alias, so both of its options retain
    # nothing while the other three still choose between 10 bytes and none
    first = program.task_alternative_groups[0]
    settled = replace(
        first,
        options=(
            replace(first.options[0], retained_alias_group_ids=()),
            first.options[1],
        ),
    )
    program = replace(
        program,
        task_alternative_groups=(settled, *program.task_alternative_groups[1:]),
    )

    options = CostedAlternatives.from_program(program)
    assert options.flexible_count == 3
    # forced to the cheaper of two options that keep the same bytes
    assert options.groups[0].forced_index == 0
    assert all(group.forced_index is None for group in options.groups[1:])

    # and the resolutions never recompute it, at any share
    for selections in _option_ids(program):
        assert selections[0] == "save"


def test_the_default_resolution_options_are_every_quarter() -> None:
    """The default is named, not implied: a caller wanting it asks for it."""

    assert tuple(Fraction(n, 4) for n in range(5)) == DEFAULT_RESOLUTION_OPTIONS
    # spelling the default out, in any order and any accepted form, is the
    # same question as naming the constant
    spelled = ("1", "3/4", "1/2", "1/4", 0)
    assert validate_resolution_options(spelled) == DEFAULT_RESOLUTION_OPTIONS
    assert _option_ids(_binary_program(64), DEFAULT_RESOLUTION_OPTIONS) == _option_ids(
        _binary_program(64)
    )


def test_a_caller_names_its_own_resolution_options() -> None:
    program = _binary_program(64)

    eighths = _option_ids(program, tuple(f"{n}/8" for n in range(9)))

    assert tuple(item.count("recompute") for item in eighths) == tuple(range(0, 65, 8))
    # order and repetition are the caller's spelling, not part of the options
    assert _option_ids(program, (1, "1/2", Fraction(1, 2), 0)) == _option_ids(
        program, ("0", "1/2", "1")
    )
    # a small inventory stays exhaustive whatever options are named
    assert len(_option_ids(_binary_program(6), ("1",))) == 64


def test_resolution_options_are_validated() -> None:
    assert validate_resolution_options(("1", "1/2", "1/2", 0)) == (
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
    )
    with pytest.raises(ValueError, match="outside"):
        validate_resolution_options(("9/8",))
    with pytest.raises(ValueError, match="Fractions, integers or strings"):
        validate_resolution_options((0.5,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Fractions, integers or strings"):
        validate_resolution_options((True,))
    with pytest.raises(ValueError, match="at least one"):
        validate_resolution_options(())
    with pytest.raises(ValueError, match="not a fraction"):
        validate_resolution_options(("half",))
    with pytest.raises(ValueError, match="not one string"):
        validate_resolution_options("1/2")
    with pytest.raises(ValueError, match="outside"):
        resolutions(_binary_program(64), ("2",))


def _ladder_program(stages: int) -> ShadowSpillProgram:
    """A chain of save/recompute stages, each shaped like the one-group fixture."""

    aliases: list[AliasGroupSpec] = []
    objects: list[ObjectSpec] = []
    tasks: list[TaskSpec] = []
    groups: list[TaskAlternativeGroup] = []
    for index in range(stages):
        aliases.extend(
            (
                AliasGroupSpec(f"input_{index}", "cuda_0", 10),
                AliasGroupSpec(f"activation_{index}", "cuda_0", 100),
                AliasGroupSpec(f"temporary_{index}", "cuda_0", 100),
            )
        )
        objects.extend(
            (
                ObjectSpec(f"in_{index}", f"input_{index}", 0, 10),
                ObjectSpec(f"act_{index}", f"activation_{index}", 0, 100),
                ObjectSpec(f"tmp_{index}", f"temporary_{index}", 0, 100),
            )
        )
        after = () if index == 0 else (f"consume_{index - 1}",)
        tasks.extend(
            (
                TaskSpec(
                    f"forward_save_{index}",
                    COMPUTE,
                    "forward_profile",
                    dependencies=after,
                    inputs=(f"in_{index}",),
                    outputs=(f"act_{index}",),
                ),
                TaskSpec(
                    f"middle_{index}",
                    COMPUTE,
                    "middle_profile",
                    dependencies=(f"forward_save_{index}",),
                    outputs=(f"tmp_{index}",),
                ),
                TaskSpec(
                    f"forward_recompute_{index}",
                    COMPUTE,
                    "forward_profile",
                    dependencies=(f"middle_{index}",),
                    inputs=(f"in_{index}",),
                    outputs=(f"act_{index}",),
                ),
                TaskSpec(
                    f"consume_{index}",
                    COMPUTE,
                    "consume_profile",
                    dependencies=(
                        f"forward_save_{index}",
                        f"middle_{index}",
                        f"forward_recompute_{index}",
                    ),
                    inputs=(f"act_{index}",),
                ),
            )
        )
        groups.append(
            TaskAlternativeGroup(
                f"tradeoff_{index}",
                (
                    TaskAlternativeOption(
                        "save", (f"forward_save_{index}",), (f"activation_{index}",)
                    ),
                    TaskAlternativeOption("recompute", (f"forward_recompute_{index}",)),
                ),
            )
        )
    return ShadowSpillProgram(
        devices=(DEVICE,),
        alias_groups=tuple(aliases),
        objects=tuple(objects),
        profiles=(
            TaskProfile("forward_profile", 100, 0, "forward_abi"),
            TaskProfile("middle_profile", 1_000, 0, "middle_abi"),
            TaskProfile("consume_profile", 100, 0, "consume_abi"),
        ),
        tasks=tuple(tasks),
        task_alternative_groups=tuple(groups),
    )


def test_resolution_options_change_no_rung_and_a_superset_is_never_worse() -> None:
    """Named eighths contain the default quarter rungs and cannot do worse.

    Deterministic mode makes every rung's outcome a function of the rung
    alone, so a rung answers the same whichever options it is planned among,
    and the best of a superset is never worse than the best of its subset.
    """

    stages = 8
    program = _ladder_program(stages)
    initial = tuple(
        ResidencySpec(f"input_{index}", MemoryLocation.DEVICE)
        for index in range(stages)
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=260,
        spill_capacity_bytes=10_000,
        fetch_bandwidth_bytes_per_second=8_000_000,
        evict_bandwidth_bytes_per_second=8_000_000,
    )
    options = GenericPlanningOptions(
        deterministic=True, minimum_object_bytes_evict_eligible=0
    )

    def by_rung(result: object) -> dict[str, int | None]:
        return {
            problem.selection_id: problem.selected_makespan_ns
            for problem in result.diagnostics.resolved_programs  # type: ignore[attr-defined]
        }

    quarters = pressurefit(
        program, initial_residency=initial, config=config, generic=options
    )
    eighths = PressureFit(
        PressureFitOptions(resolution_options=tuple(f"{n}/8" for n in range(9)))
    )(program, initial_residency=initial, config=config, generic=options)

    assert len(by_rung(eighths)) == 9
    assert len(by_rung(quarters)) == 5
    assert set(by_rung(quarters)) <= set(by_rung(eighths))
    assert all(
        by_rung(eighths)[rung] == value for rung, value in by_rung(quarters).items()
    )
    assert eighths.simulation.makespan_ns <= quarters.simulation.makespan_ns
