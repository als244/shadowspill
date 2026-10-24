"""Project allocation events to physical placement lifetimes."""

from shadowspill.planner.admission import TaskAllocationStep, TaskAllocationStepKind
from shadowspill.task.allocations import TaskAllocationEvent, TaskAllocationOperation

from .outputs import TaskOutputBinding


def _task_allocation_steps(
    task_id: str,
    trace: tuple[TaskAllocationEvent, ...],
    bindings: tuple[TaskOutputBinding, ...],
    *,
    persistent_aliases: set[str],
) -> tuple[TaskAllocationStep, ...]:
    """Project one physical profile without making it a runtime contract.

    The profiled order is used only by offline dynamic-pool admission.  Output
    leaves that do not become program objects are released at task completion;
    persistent output allocations remain live for ownership publication.
    """

    alias_by_leaf = {item.leaf_index: item.alias_group_id for item in bindings}
    live: dict[int, str | None] = {}
    steps: list[TaskAllocationStep] = []
    observed_aliases: set[str] = set()
    for event in trace:
        ordinal = event.allocation_ordinal
        if event.operation is TaskAllocationOperation.ALLOCATE:
            aliases = {
                alias_by_leaf[leaf]
                for leaf in event.output_leaf_indices
                if leaf in alias_by_leaf
            }
            if len(aliases) > 1:
                raise ValueError(
                    f"task {task_id} allocation {ordinal} backs distinct "
                    f"persistent aliases {sorted(aliases)!r}"
                )
            alias_id = next(iter(aliases), None)
            if alias_id is not None:
                observed_aliases.add(alias_id)
            live[ordinal] = alias_id
            steps.append(
                TaskAllocationStep(
                    ordinal,
                    TaskAllocationStepKind.ALLOCATE,
                    event.charged_bytes,
                    alias_id,
                    event.reuses_ordinal,
                )
            )
            continue
        if ordinal not in live:
            raise ValueError(
                f"task {task_id} allocation trace releases unknown ordinal {ordinal}"
            )
        del live[ordinal]
        steps.append(TaskAllocationStep(ordinal, TaskAllocationStepKind.RELEASE))
    missing = persistent_aliases - observed_aliases
    unexpected = observed_aliases - persistent_aliases
    if missing or unexpected:
        raise ValueError(
            f"task {task_id} physical output bindings disagree with ShadowSpillProgram "
            f"ownership: missing={sorted(missing)!r}, "
            f"unexpected={sorted(unexpected)!r}"
        )
    # The profiler intentionally omits logical frees for returned tensors.
    # Unselected returned leaves are ordinary task-local allocations and end
    # at the task boundary; declared ShadowSpillProgram outputs remain live.
    steps.extend(
        TaskAllocationStep(ordinal, TaskAllocationStepKind.RELEASE)
        for ordinal, alias_id in live.items()
        if alias_id is None
    )
    return tuple(steps)


def _anonymous_peak_extents(
    steps: tuple[TaskAllocationStep, ...],
) -> tuple[int, ...]:
    """Return the exact anonymous live-set peak encoded by one task trace."""

    return _workspace_peak_extents(steps, transient_aliases=set())


def _workspace_peak_extents(
    steps: tuple[TaskAllocationStep, ...],
    *,
    transient_aliases: set[str],
) -> tuple[int, ...]:
    """Return the live-set peak not charged as fresh program output storage."""

    live: dict[int, int] = {}
    peak: tuple[int, ...] = ()
    live_bytes = 0
    peak_bytes = 0
    for step in steps:
        if step.kind is TaskAllocationStepKind.ALLOCATE:
            if (
                step.output_alias_group_id is None
                or step.output_alias_group_id in transient_aliases
            ):
                live[step.allocation_ordinal] = step.charged_bytes
                live_bytes += step.charged_bytes
        else:
            live_bytes -= live.pop(step.allocation_ordinal, 0)
        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak = tuple(sorted(live.values()))
    return peak
