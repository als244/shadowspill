"""The facts' tasks as columns: what they compile to, and what they name."""

from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.planner.admission import (
    AdmissionFacts,
    StorageHandoff,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.planner.admission.indexing import _compiled_tasks
from shadowspill.simulator.capi import NO_INDEX

ALLOCATE = TaskAllocationStepKind.ALLOCATE
RELEASE = TaskAllocationStepKind.RELEASE


def _task(rng: random.Random, task_id: str, aliases: list[str]) -> TaskAdmissionSpec:
    """A valid task: allocations under scattered ordinals, some reusing a
    retired one's bytes, some becoming outputs, every anonymous one released;
    handoffs between aliases it does not output."""

    if rng.random() < 0.2:
        return TaskAdmissionSpec(task_id)
    ordinals = rng.sample(range(5_000), rng.randint(1, 12))
    outputs = rng.sample(aliases, min(len(aliases), rng.randint(0, 3)))
    steps: list[TaskAllocationStep] = []
    live: dict[int, int] = {}
    retired: dict[int, int] = {}
    persistent: set[int] = set()
    unbound = list(outputs)
    for ordinal in ordinals:
        while live.keys() - persistent and rng.random() < 0.4:
            released = rng.choice(sorted(live.keys() - persistent))
            steps.append(TaskAllocationStep(released, RELEASE))
            retired[released] = live.pop(released)
        reusable = sorted(retired)
        if reusable and rng.random() < 0.5:
            source = rng.choice(reusable)
            charged = retired.pop(source)
        else:
            source, charged = None, rng.randint(1, 1 << 20)
        output = unbound.pop() if unbound and rng.random() < 0.5 else None
        steps.append(
            TaskAllocationStep(
                ordinal,
                ALLOCATE,
                charged,
                output_alias_group_id=output,
                reuses_allocation_ordinal=source,
            )
        )
        live[ordinal] = charged
        if output is not None:
            persistent.add(ordinal)
    for ordinal in sorted(live.keys() - persistent):
        steps.append(TaskAllocationStep(ordinal, RELEASE))
    bound = [step.output_alias_group_id for step in steps if step.output_alias_group_id]
    split = rng.randint(0, len(bound))
    others = [alias for alias in aliases if alias not in bound]
    pairs = rng.sample(others, min(len(others) // 2 * 2, 2 * rng.randint(0, 2)))
    return TaskAdmissionSpec(
        task_id,
        workspace_extents=tuple(rng.randint(1, 4096) for _ in range(rng.randint(0, 3))),
        fresh_output_aliases=tuple(bound[:split]),
        replacement_aliases=tuple(bound[split:]),
        storage_handoffs=tuple(
            StorageHandoff(pairs[index], pairs[index + 1])
            for index in range(0, len(pairs), 2)
        ),
        allocation_steps=tuple(steps),
    )


def _facts(seed: int, count: int) -> AdmissionFacts:
    rng = random.Random(seed)
    aliases = [f"alias_{index:04d}" for index in range(3 * count)]
    return AdmissionFacts(
        "cuda_0",
        1 << 30,
        1 << 30,
        256,
        tuple(_task(rng, f"task_{index:04d}", aliases) for index in range(count)),
    )


def _template(facts: AdmissionFacts, seed: int) -> SimpleNamespace:
    """What the compile reads of a simulation template: some of the tasks, in
    an order of its own, and every alias at a position of its own."""

    rng = random.Random(seed)
    task_ids = [task.task_id for task in facts.tasks]
    selected = rng.sample(task_ids, max(1, len(task_ids) * 3 // 4))
    aliases = sorted(
        {name for task in facts.tasks for name in _aliases_named(task)}
        | {f"unused_{index}" for index in range(5)}
    )
    rng.shuffle(aliases)
    return SimpleNamespace(
        task_ids=tuple(selected),
        alias_index={name: index for index, name in enumerate(aliases)},
        derived={},
    )


def _aliases_named(task: TaskAdmissionSpec) -> list[str]:
    return [
        *task.fresh_output_aliases,
        *task.replacement_aliases,
        *(item.source_alias_group_id for item in task.storage_handoffs),
        *(item.destination_alias_group_id for item in task.storage_handoffs),
        *(
            step.output_alias_group_id
            for step in task.allocation_steps
            if step.output_alias_group_id
        ),
    ]


def _per_step_compile(facts: AdmissionFacts, template: SimpleNamespace) -> tuple:
    """The compile as a loop over every step: the specification the column
    compile is checked against."""

    by_task = {task.task_id: task for task in facts.tasks}
    tasks = [by_task[task_id] for task_id in template.task_ids]
    alias = template.alias_index

    def flattened(rows: list[list[int]]) -> tuple[list[int], list[int]]:
        offsets, values = [0], []
        for row in rows:
            values.extend(row)
            offsets.append(len(values))
        return offsets, values

    rows, next_slot = [], 0
    for task in tasks:
        slot_by_ordinal: dict[int, int] = {}
        row = []
        for step in task.allocation_steps:
            if step.kind is ALLOCATE:
                if step.reuses_allocation_ordinal is None:
                    slot, next_slot = next_slot, next_slot + 1
                else:
                    slot = slot_by_ordinal[step.reuses_allocation_ordinal]
                slot_by_ordinal[step.allocation_ordinal] = slot
            else:
                slot = slot_by_ordinal[step.allocation_ordinal]
            row.append(
                (
                    0 if step.kind is ALLOCATE else 1,
                    slot,
                    step.charged_bytes,
                    NO_INDEX
                    if step.output_alias_group_id is None
                    else alias[step.output_alias_group_id],
                )
            )
        rows.append(row)
    workspace = flattened([list(task.workspace_extents) for task in tasks])
    fresh = flattened([[alias[a] for a in task.fresh_output_aliases] for task in tasks])
    replaced = flattened(
        [[alias[a] for a in task.replacement_aliases] for task in tasks]
    )
    sources = flattened(
        [
            [alias[h.source_alias_group_id] for h in task.storage_handoffs]
            for task in tasks
        ]
    )
    destinations = flattened(
        [
            [alias[h.destination_alias_group_id] for h in task.storage_handoffs]
            for task in tasks
        ]
    )
    allocation_offsets, kinds = flattened([[item[0] for item in row] for row in rows])
    _, slots = flattened([[item[1] for item in row] for row in rows])
    _, charged = flattened([[item[2] for item in row] for row in rows])
    _, outputs = flattened([[item[3] for item in row] for row in rows])
    arrays = (
        workspace[0],
        workspace[1],
        fresh[0],
        fresh[1],
        replaced[0],
        replaced[1],
        sources[0],
        sources[1],
        destinations[1],
        allocation_offsets,
        slots,
        charged,
        outputs,
        kinds,
    )
    # The library is handed at least one element per array.
    return tuple(values or [0] for values in arrays), next_slot, len(tasks)


@pytest.mark.parametrize("seed", range(40))
def test_the_column_compile_is_the_per_step_compile(seed: int) -> None:
    facts = _facts(seed, count=1 + seed % 23)
    template = _template(facts, seed)

    buffers, slot_count, task_count = _compiled_tasks(facts, template)  # type: ignore[arg-type]

    expected, expected_slots, expected_tasks = _per_step_compile(facts, template)
    assert [list(buffer) for buffer in buffers] == [list(values) for values in expected]
    assert (slot_count, task_count) == (expected_slots, expected_tasks)


def test_a_compile_without_one_of_the_templates_tasks_is_refused() -> None:
    facts = _facts(3, count=5)
    template = _template(facts, 3)
    template.task_ids = (*template.task_ids, "task_missing")
    with pytest.raises(ValueError, match="lacks selected task 'task_missing'"):
        _compiled_tasks(facts, template)  # type: ignore[arg-type]


def test_equal_tasks_have_one_digest_and_every_field_moves_it() -> None:
    facts = _facts(7, count=12)
    again = _facts(7, count=12)
    assert facts.tasks is not again.tasks
    assert facts.digest == again.digest
    # A capacity variant hashes the same tasks once.
    variant = replace(facts, pool_capacity_bytes=1 << 31)
    assert variant.task_columns is facts.task_columns
    assert variant.digest != facts.digest

    # An allocation no later step reuses, so its bytes can change alone.
    task, index = next(
        (task, index)
        for task in facts.tasks
        for index, step in enumerate(task.allocation_steps)
        if step.kind is ALLOCATE
        and step.reuses_allocation_ordinal is None
        and all(
            other.reuses_allocation_ordinal != step.allocation_ordinal
            for other in task.allocation_steps
        )
    )
    steps = list(task.allocation_steps)
    steps[index] = replace(steps[index], charged_bytes=steps[index].charged_bytes + 1)
    changes = {
        "an allocation's bytes": replace(task, allocation_steps=tuple(steps)),
        "a workspace extent": replace(
            task, workspace_extents=(*task.workspace_extents, 17)
        ),
        "the task's name": replace(task, task_id="task_renamed"),
    }
    seen = {facts.digest}
    for change, changed in changes.items():
        tasks = tuple(changed if item is task else item for item in facts.tasks)
        digest = replace(facts, tasks=tasks, _tasks_derived=[]).digest
        assert digest not in seen, change
        seen.add(digest)


def test_steps_moved_between_tasks_change_the_digest() -> None:
    """The same steps under different task boundaries are different facts."""

    step = TaskAllocationStep(0, ALLOCATE, 64)
    release = TaskAllocationStep(0, RELEASE)
    one = AdmissionFacts(
        "cuda_0",
        1024,
        1024,
        1,
        (
            TaskAdmissionSpec(
                "a", workspace_extents=(64,), allocation_steps=(step, release)
            ),
            TaskAdmissionSpec("b"),
        ),
    )
    other = AdmissionFacts(
        "cuda_0",
        1024,
        1024,
        1,
        (
            TaskAdmissionSpec("a"),
            TaskAdmissionSpec(
                "b", workspace_extents=(64,), allocation_steps=(step, release)
            ),
        ),
    )
    assert one.digest != other.digest
