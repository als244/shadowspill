"""The selected tasks compiled, the plan admitted, the callable published."""

from collections.abc import Callable
from typing import NoReturn

import torch.nn as nn

from shadowspill.pipeline.admission import (
    physical_admission,
    project_runtime_fixed_layout,
    reconcile_spill_pool,
    seal_physical_budget,
)
from shadowspill.pipeline.common import PlanningTimer
from shadowspill.pytorch.planning.admission import FixedLayoutSelection
from shadowspill.runtime import Runtime
from shadowspill.runtime.abi import INITIAL_ACTIONS_TASK_ID
from shadowspill.runtime.plan import (
    PlanMemory,
    RuntimeBridge,
)
from shadowspill.runtime.teardown import prepare_failure_cleanup

from ...callables import PlannedForward
from ...execution import ForwardExecutor
from ...materialization import (
    MaterializedForwardState,
)
from ..artifacts import (
    ForwardCaptureArtifacts,
    ForwardProfileArtifacts,
    ForwardProgramArtifacts,
)
from ..stores import PlanningStores
from .plan import _forward_execution_plan
from .programs import _build_forward_admission
from .report import _forward_plan_report


def admit_forward_plan(
    model: nn.Module,
    captured: ForwardCaptureArtifacts,
    profiled: ForwardProfileArtifacts,
    program: ForwardProgramArtifacts,
    selection: FixedLayoutSelection,
    *,
    memory: PlanMemory,
    stores: PlanningStores,
    timer: PlanningTimer,
    started: int,
) -> PlannedForward:
    """Physically admit a selection and publish the executable callable/report."""

    selected = selection.result
    with timer.measure("spill_admission"):
        reconcile_spill_pool(
            predicted_peak=selected.simulation.spill_peak_bytes,
            budget=memory.spill_budget,
        )
    selected_admission = _build_forward_admission(
        program,
        selection,
        timer,
    )
    admission = physical_admission(
        memory,
        captured.installed,
        workspace_reserve=program.workspace_reserve,
        predicted_spill_peak_bytes=selected.simulation.spill_peak_bytes,
        predicted_fragmentation_bytes=(
            selected_admission.predicted_fragmentation_bytes
        ),
    )
    admitted_result = selected_admission.apply_prediction(selected)
    execution_plan = _forward_execution_plan(
        program.lowered,
        admitted_result,
        admission,
    )
    fixed_layout = selected_admission.fixed_layout
    if fixed_layout is None:
        raise AssertionError("forward admission did not produce a fixed layout")
    runtime_fixed_layout = project_runtime_fixed_layout(
        fixed_layout,
        execution_plan.program,
        execution_plan.schedule,
        initial_task_id=INITIAL_ACTIONS_TASK_ID,
        dynamic_task_allocations=(selected_admission.dynamic_provider_allocations()),
    )
    bridge = RuntimeBridge(
        memory.runtime,
        execution_plan.program,
        memory.plan_handle,
        execution_pool_id=memory.execution.pool_id,
        spill_pool_id=memory.spill.pool_id,
    )
    state: MaterializedForwardState | None = None
    try:
        with timer.measure("materialization"):
            state = MaterializedForwardState(
                model,
                program.lowered,
                captured.capture,
                captured.cpu_inputs,
                bridge,
                runtime=memory.runtime,
                device_ordinal=captured.device_ordinal,
                shared_inputs=captured.shared_inputs,
            )
        with timer.measure("physical_sealing"):
            seal_physical_budget(
                captured.installed,
                execution_plan,
                fixed_layout,
            )
        with timer.measure("callable_construction"):
            executor = ForwardExecutor(
                captured.partitioned,
                program.lowered,
                execution_plan,
                bridge,
                state,
                profiled.compiled_tasks.functions,
                captured.capture.user_output_indices,
                captured.output_tree_spec,
                shared_outputs=captured.shared_outputs,
                fixed_layout=runtime_fixed_layout,
                memory_envelopes=selected_admission.envelopes_by_task(),
            )
        report = _forward_plan_report(
            model,
            captured,
            profiled,
            program,
            selection,
            selected_admission,
            admitted_result,
            execution_plan,
            stores=stores,
            memory=memory,
            timer=timer,
            started=started,
        )
        return PlannedForward(
            model,
            captured.signature,
            executor,
            state,
            report,
            memory.runtime,
            memory.plan_handle,
        )
    except BaseException as error:
        if state is not None:
            _rollback_forward_failure(
                memory.runtime,
                error,
                state.restore_cpu_and_unregister,
                operation="admit forward plan",
            )
        raise


def _rollback_forward_failure(
    runtime: Runtime,
    error: BaseException,
    rollback: Callable[[], None],
    *,
    operation: str,
) -> NoReturn:
    """Recover a planning OOM before releasing materialized frontend state."""

    prepare_failure_cleanup(
        runtime,
        error,
        operation=operation,
        synchronize_unlatched=False,
    )
    try:
        rollback()
    except BaseException as cleanup_error:
        error.add_note(
            f"Failed to roll back materialized forward state: {cleanup_error}"
        )
    raise error
