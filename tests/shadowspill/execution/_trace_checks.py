"""Reconcile a real invocation against its saved schedule and measured profiles."""


def check_trace(plan, trace):
    assert trace is not None and trace.summary.trace_complete
    runtime, summary = trace.runtime, trace.summary
    assert not runtime.event_overflow and not runtime.allocation_event_overflow
    assert 0 < runtime.began_at_ns <= runtime.ended_at_ns
    assert runtime.materialized_allocation_requests == (
        runtime.allocation_requests - runtime.zero_byte_allocation_requests
    )
    assert {
        "session_begin",
        "session_end",
        "before_task",
        "after_task",
        "action_queued",
    } <= {event.kind.name.lower() for event in runtime.events}
    assert summary.profiled_task_seconds > 0 and summary.real_task_event_seconds > 0
    assert trace.allocator.live_allocations_before >= 0
    assert trace.allocator.live_allocations_after >= 0

    selected = plan._costed_program.selected_tasks(plan.selections)
    assert {row.task_id for row in trace.tasks.values()} == {
        task.task_id for task in selected
    }
    compute = {key: row for key, row in trace.tasks.items() if row.phase != "control"}
    assert set(trace.timelines.compute) == compute.keys()
    assert {row.phase for row in summary.phase_comparisons} == {
        row.phase for row in compute.values()
    }
    for key, row in compute.items():
        expected = selected[row.execution_ordinal]
        assert row.execution_task_id == key == f"execution_{row.execution_ordinal:06d}"
        assert row.task_id == expected.task_id and row.semantic_name
        assert row.expected_profile_seconds > 0
        assert row.simulated_finished_at_seconds >= row.simulated_started_at_seconds
        assert (
            row.compute_finished_at_seconds
            > row.compute_started_at_seconds
            >= row.compute_reached_at_seconds
            >= 0
        )
        assert (
            row.before_task_entered_at_seconds
            <= row.before_task_exited_at_seconds
            <= row.after_task_entered_at_seconds
            <= row.after_task_exited_at_seconds
        )

    count = 0
    for lane in (trace.timelines.fetch, trace.timelines.evict):
        records = getattr(trace.transfers, lane.summary.direction)
        assert set(lane.order) == records.keys()
        assert lane.summary.measured_transfers == len(records)
        count += len(records)
        for index, key in enumerate(lane.order):
            row = records[key]
            assert row.sequence == index and row.direction == lane.summary.direction
            assert row.bytes > 0 and row.triggered_by in trace.tasks
            assert row.simulated_started_at_seconds is not None
            assert row.completion_observed_at_seconds >= row.dispatched_at_seconds
            assert row.lane_finished_at_seconds >= row.lane_started_at_seconds
    assert count == len(plan.timeline.transfer_intervals)
