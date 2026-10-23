"""Collective records retain separate issue/GPU clocks and ownership links."""

import json

from shadowspill.pytorch.execution.collectives import decode_collectives


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


def test_host_copy_phase_retains_bytes_ownership_and_separate_completeness():
    captured = decode_collectives(
        {
            "records": [],
            "host_copies": {
                "capacity": 2,
                "dropped": 1,
                "records": [
                    {
                        "source": "fetch",
                        "bytes": 128,
                        "task_id": 9,
                        "object_id": 4001,
                        "action_ordinal": 3,
                        "issued_at_ns": 100,
                        "started_at_ns": 200,
                        "finished_at_ns": 300,
                    }
                ],
            },
        },
        groups={},
        tasks={9: ("task_000009", "execution_000012")},
        aliases={4001: "alias_000002"},
    )
    assert captured.complete  # Collective capture is independently complete.
    copies = captured.as_dict()["host_copies"]
    assert not copies["complete"] and copies["dropped"] == 1
    record = copies["records"][0]
    assert record["bytes"] == 128
    assert record["execution_task_id"] == "execution_000012"
    assert record["alias_group_id"] == "alias_000002"
    assert record["finished_at_ns"] - record["started_at_ns"] == 100
    assert decode_collectives({}, groups={}, tasks={}, aliases={}).host_copies is None
