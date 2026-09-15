"""A plan run admitted to the runtime, once, before it executes."""

from __future__ import annotations

from dataclasses import replace

from shadowspill.ir import MemoryAction, MemoryActionKind
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    admit_caller_acquisition,
    admit_fixed_layout,
    admit_initial_actions,
    admit_task,
    seal_fixed_layout,
)
from shadowspill.runtime.fixed_layout import RuntimeFixedLayout
from shadowspill.runtime.transfer_labels import TransferLabelIndex

from ..records import (
    ExecutionTaskRecord as _ExecutionTaskRecord,
)
from ..records import (
    PlanRun as _PlanRun,
)


def admit_run(
    bridge: RuntimeBridge,
    run: _PlanRun,
    fixed_layout: RuntimeFixedLayout,
) -> _PlanRun:
    """Admit one plan run's fixed layout, initial actions, tasks and caller
    acquisitions to the runtime, and seal the layout."""

    admit_fixed_layout(bridge, fixed_layout)
    initial_actions = tuple(
        MemoryAction("task_000000", alias_id, MemoryActionKind.FETCH)
        for alias_id in run.initial_fetches
    )
    admit_initial_actions(
        bridge,
        initial_actions,
        task_number=fixed_layout.initial_task_id,
        action_trace_labels=tuple(
            f"shadowspill.fetch.initial.{alias_id}" for alias_id in run.initial_fetches
        ),
    )
    labels = TransferLabelIndex(
        run.plan.program,
        {record.task.task_id: record.trace_label for record in run.execution},
    )
    admitted: list[_ExecutionTaskRecord] = []
    for record in run.execution:
        admitted.append(
            replace(
                record,
                task_handle=admit_task(
                    bridge,
                    record.task,
                    record.input_aliases,
                    record.actions,
                    labels.labels_for(record.actions),
                    record.memory_envelope,
                    trace_label=record.trace_label,
                    publications=record.publications,
                ),
            )
        )
    caller_aliases = tuple(
        alias_id for values in run.public_by_microbatch for alias_id in values
    )
    caller_acquisition_handle = admit_caller_acquisition(bridge, caller_aliases)
    seal_fixed_layout(bridge)
    return replace(
        run,
        execution=tuple(admitted),
        initial_task_id=fixed_layout.initial_task_id,
        caller_acquisition_handle=caller_acquisition_handle,
    )


__all__ = ["admit_run"]
