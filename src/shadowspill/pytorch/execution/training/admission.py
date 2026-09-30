"""A plan run admitted to the runtime, once, before it executes."""

from __future__ import annotations

from dataclasses import replace

from shadowspill.runtime.fixed_layout import RuntimeFixedLayout
from shadowspill.runtime.plan import (
    RuntimeBridge,
    admit_caller_acquisition,
    admit_fixed_layout,
    admit_task,
    seal_fixed_layout,
)
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
    """Admit the layout, scheduled tasks and caller acquisitions, then seal."""

    admit_fixed_layout(bridge, fixed_layout)
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
    caller_aliases = run.public_aliases
    caller_acquisition_handle = admit_caller_acquisition(bridge, caller_aliases)
    seal_fixed_layout(bridge)
    return replace(
        run,
        execution=tuple(admitted),
        caller_acquisition_handle=caller_acquisition_handle,
    )


__all__ = ["admit_run"]
