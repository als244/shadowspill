"""Resolve an opt-in execution trace into immutable diagnostics."""

from shadowspill.diagnostics.step import (
    StepDiagnostics,
    Timelines,
    TransferQueue,
    TransferRecords,
)
from shadowspill.diagnostics.timing import ArmedExecutionTiming
from shadowspill.planner.diagnostics.mapping import FrozenMapping
from shadowspill.runtime.plan.bridge import RuntimeBridge

from .compute import compute_record, selected_intervals, stream_task
from .evidence import (
    build_allocator_trace,
    build_runtime_trace,
    resolve_trace_evidence,
    validate_completed_timing,
)
from .summary import build_step_summary
from .transfers import transfer_lanes


def collect_step_diagnostics(
    timing: ArmedExecutionTiming,
    bridge: RuntimeBridge,
) -> StepDiagnostics:
    """Synchronize and assemble all evidence for one traced real step."""

    validate_completed_timing(timing)
    evidence = resolve_trace_evidence(timing, bridge)
    simulation = timing.simulation
    if simulation is None:
        raise RuntimeError("execution trace omitted selected simulator evidence")
    stream_tasks = tuple(stream_task(timing, task_id) for task_id in timing.task_order)
    intervals = selected_intervals(simulation, stream_tasks)
    origin_ns = timing.origin_recorded_ns
    compute = tuple(
        compute_record(item, intervals[item.task_id], origin_ns)
        for item in stream_tasks
    )
    lanes = transfer_lanes(timing, simulation, evidence, bridge, origin_ns)
    model_tasks = tuple(item for item in compute if item.phase != "control")
    runtime = build_runtime_trace(evidence)
    transfers = TransferRecords(
        fetch=FrozenMapping(
            {item.transfer_id: item for item in lanes["fetch"].records}
        ),
        evict=FrozenMapping(
            {item.transfer_id: item for item in lanes["evict"].records}
        ),
    )
    timelines = Timelines(
        host_origin_ns=origin_ns,
        first_task_started_at_seconds=min(
            item.compute_started_at_seconds for item in model_tasks
        ),
        compute=tuple(item.execution_task_id for item in model_tasks),
        fetch=TransferQueue(
            order=tuple(item.transfer_id for item in lanes["fetch"].records),
            summary=lanes["fetch"].summary,
        ),
        evict=TransferQueue(
            order=tuple(item.transfer_id for item in lanes["evict"].records),
            summary=lanes["evict"].summary,
        ),
    )
    return StepDiagnostics(
        summary=build_step_summary(
            timing, simulation, model_tasks, intervals, runtime, lanes
        ),
        tasks=FrozenMapping({item.execution_task_id: item for item in compute}),
        transfers=transfers,
        timelines=timelines,
        allocator=build_allocator_trace(evidence),
        runtime=runtime,
        collectives=timing.collectives,
    )
