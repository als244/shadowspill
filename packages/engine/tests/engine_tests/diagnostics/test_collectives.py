"""Collective records retain separate issue/GPU clocks and ownership links."""

import json
from types import SimpleNamespace

from shadowspill.diagnostics.collection.transfers import transfer_copies
from shadowspill.execution.recording.collectives import decode_collectives
from shadowspill.runtime.trace import RuntimeTraceEvent, RuntimeTraceEventKind


def test_collective_phase_preserves_issue_queue_delay_and_group_identity():
    captured = decode_collectives(
        {
            "capacity": 16,
            "records": [
                {
                    "operation": "_reduce_scatter_base_",
                    "group": "local-2",
                    "group_rank": 1,
                    "group_size": 2,
                    "group_sequence": 23,
                    "input_bytes": 1024,
                    "output_bytes": 512,
                    "source": "evict",
                    "task_id": 9,
                    "object_id": 4001,
                    "action_ordinal": 3,
                    "issued_at_ns": 100,
                    "started_at_ns": 900,
                    "finished_at_ns": 1000,
                }
            ],
        },
        groups={"local-2": (2, 3)},
        tasks={9: ("task_000009", "execution_000012")},
        aliases={4001: "alias_000002"},
    )
    assert captured.complete
    record = captured.records[0]
    assert record.operation == "reduce_scatter"
    assert record.group_ranks == (2, 3)
    assert record.group_sequence == 23
    assert record.execution_task_id == "execution_000012"
    assert record.alias_group_id == "alias_000002"
    assert record.started_at_ns - record.issued_at_ns == 800
    assert record.finished_at_ns - record.started_at_ns == 100
    assert (
        json.loads(json.dumps(captured.as_dict()))["records"][0]["output_bytes"] == 512
    )


def test_dropped_or_unmeasured_collectives_make_trace_incomplete():
    for failure in ("dropped", "errors", "missing_timings"):
        captured = decode_collectives(
            {"capacity": 1, failure: 1, "records": []},
            groups={},
            tasks={},
            aliases={},
        )
        assert not captured.complete
        assert not captured.as_dict()["complete"]


def _event(
    kind,
    sequence,
    *,
    task=9,
    obj=4001,
    size=128,
    direction=0,
    ordinal=3,
    lane=(100, 200, 300),
):
    return RuntimeTraceEvent(
        sequence=sequence,
        timestamp_ns=1_000 + sequence,
        step_id=1,
        task_id=task,
        object_id=obj,
        allocation_id=None,
        bytes=size,
        kind=kind,
        detail_0=direction,
        detail_1=ordinal,
        lane_issued_at_ns=lane[0],
        lane_started_at_ns=lane[1],
        lane_finished_at_ns=lane[2],
    )


def _traced(events, *, overflow=False):
    timing = SimpleNamespace(
        tasks={"task_000009": SimpleNamespace(execution_ordinal=12)},
        alias_accesses={"alias_000002": (), "alias_000003": ()},
    )
    objects = SimpleNamespace(
        runtime_object_id={"alias_000002": 4001, "alias_000003": 4002}.__getitem__,
        requires_storage=lambda alias: alias == "alias_000002",
    )
    evidence = SimpleNamespace(
        runtime_trace=SimpleNamespace(event_overflow=overflow, events=tuple(events))
    )
    return timing, evidence, SimpleNamespace(objects=objects)


def test_a_transformed_transfers_copy_is_the_interval_its_lane_marked():
    kinds = RuntimeTraceEventKind
    events = [
        _event(kinds.TRANSFER_DISPATCHED, 0, lane=(None, None, None)),
        # The copy alone, then the transfer as a whole: copy, then a collective.
        _event(kinds.TRANSFER_COPIED, 1, size=128, lane=(100, 200, 300)),
        _event(kinds.TRANSFER_COMPLETED, 2, size=128, ordinal=7, lane=(90, 150, 900)),
        _event(
            kinds.TRANSFER_COPIED, 3, direction=1, size=64, lane=(1_000, 1_100, 1_150)
        ),
    ]
    copies = transfer_copies(*_traced(events))
    first, second = copies.as_dict()["records"]
    assert first["copy_id"] == "host_copy_000000" and first["source"] == "fetch"
    assert first["bytes"] == 128 and first["action_ordinal"] == 3
    assert first["task_id"] == "task_000009"
    assert first["execution_task_id"] == "execution_000012"
    assert first["object_id"] == 4001 and first["alias_group_id"] == "alias_000002"
    assert first["started_at_ns"] - first["issued_at_ns"] == 100
    assert first["finished_at_ns"] - first["started_at_ns"] == 100
    assert second["source"] == "evict" and second["bytes"] == 64
    assert second["finished_at_ns"] - second["started_at_ns"] == 50
    # A step with no such transfer has an empty ledger, not a missing one.
    assert transfer_copies(*_traced(events[:1])).records == ()
    # An overflowed trace holds only the step's first events and joins nothing.
    assert transfer_copies(*_traced(events, overflow=True)) is None
    assert decode_collectives({}, groups={}, tasks={}, aliases={}).host_copies is None
