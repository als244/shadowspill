"""Read sequential forward/backward passes without model or stage identities."""

from shadowspill.ir import ShadowSpillProgram


def alternative_traversals(
    program: ShadowSpillProgram, flexible: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Return flexible group indices in each pass's forward task order.

    Required groups are absent from ``flexible``, so a paired terminal loss
    does not split its enclosing pass. Interleaved sweeps and unlabelled groups
    have no unambiguous per-pass tail and are refused when a caller asks for one.
    """
    tasks = {task.task_id: task for task in program.tasks}
    positions = {task.task_id: i for i, task in enumerate(program.tasks)}
    events: list[tuple[int, str, int]] = []
    for index in flexible:
        group = program.task_alternative_groups[index]
        phases = {
            phase: [
                positions[task_id]
                for option in group.options
                for task_id in option.active_task_ids
                if tasks[task_id].phase == phase
            ]
            for phase in ("forward", "backward")
        }
        if not all(phases.values()):
            raise ValueError(
                "tail saves require forward/backward task phases for each flexible "
                f"group; {group.group_id!r} lacks one; use tail_save_counts=(0,) "
                "or a custom search for a different traversal"
            )
        forward, backward = max(phases["forward"]), min(phases["backward"])
        if forward >= backward:
            raise ValueError("tail saves require forward before backward in each group")
        events.extend(((forward, "forward", index), (backward, "backward", index)))
    passes: list[list[int]] = []
    pending: set[int] = set()
    backward_started = False
    for _, phase, index in sorted(events):
        if phase == "forward":
            if backward_started and pending:
                raise ValueError(
                    "tail saves require non-interleaved forward/backward passes; "
                    "use tail_save_counts=(0,) or a custom search for this traversal"
                )
            if not pending:
                passes.append([])
                backward_started = False
            passes[-1].append(index)
            pending.add(index)
        else:
            pending.remove(index)
            backward_started = True
    return tuple(tuple(group) for group in passes)
