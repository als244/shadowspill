"""Composite transfers must not masquerade as physical memcpy measurements."""

import copy

import pytest

from shadowspill.diagnostics.transfer_phases import transfer_phase_data


def fixture():
    program = {
        "alias_groups": [
            {
                "alias_group_id": "a",
                "size_bytes": 800,
                "transfer": {"spill_size_bytes": 100},
            }
        ]
    }
    collective = {
        "source": "fetch",
        "operation": "all_gather",
        "alias_group_id": "a",
        "execution_task_id": "e0",
        "started_at_ns": 400,
        "finished_at_ns": 900,
        "input_bytes": 100,
        "output_bytes": 800,
        "collective_id": "c0",
        "group": "dp",
        "group_size": 8,
        "group_sequence": 3,
    }
    transfer = {
        "transfer_id": "fetch_0",
        "alias_group_id": "a",
        "triggered_by": "e0",
        "bytes": 100,
        "lane": {
            "lane_started_at_seconds": 100e-9,
            "lane_finished_at_seconds": 1000e-9,
        },
    }
    diagnostics = {
        "collectives": {"enabled": True, "complete": True, "records": [collective]},
        "transfers": {"fetch": {"fetch_0": transfer}, "evict": {}},
    }
    return program, diagnostics


def test_old_composite_trace_leaves_copy_and_dispatch_time_unresolved():
    program, diagnostics = fixture()
    result = transfer_phase_data(program, diagnostics)
    assert [(p["kind"], p["start_ns"], p["end_ns"]) for p in result["fetch"]] == [
        ("unresolved", 100, 400),
        ("collective", 400, 900),
        ("unresolved", 900, 1000),
    ]
    assert all(p["bytes"] is None for p in result["fetch"] if p["kind"] == "unresolved")
    assert result["unresolved_host_bytes"] == {"fetch": 100, "evict": 0}
    assert result["summary"][0]["input_bytes"] == 100
    assert result["summary"][0]["output_bytes"] == 800
    assert result["summary"][0]["busy_ns"] == 500
    assert result["summary"][0]["host_copy_gbps"] is None


def test_measured_copy_phase_has_local_bytes_and_its_own_duration():
    program, diagnostics = fixture()
    diagnostics["collectives"]["host_copies"] = {
        "complete": True,
        "records": [
            {
                "source": "fetch",
                "alias_group_id": "a",
                "execution_task_id": "e0",
                "started_at_ns": 110,
                "finished_at_ns": 310,
                "bytes": 100,
                "copy_id": "h0",
            }
        ],
    }
    before = copy.deepcopy(diagnostics)
    result = transfer_phase_data(program, diagnostics)
    assert diagnostics == before
    assert result["unmatched_records"] == 0
    assert result["unresolved_copy_transfers"]["fetch"] == 0
    copy_row = next(r for r in result["summary"] if r["copy_bytes"] is not None)
    assert copy_row["host_copy_gbps"] == 0.5  # 100 physical bytes / 200 ns
    assert sum(p["end_ns"] - p["start_ns"] for p in result["fetch"]) == 900
    measured = next(p for p in result["fetch"] if p["kind"] == "copy")
    assert measured["bytes"] == measured["host_bytes"] == 100
    assert measured["execution_bytes"] == 800
    assert all(
        p["operation"] == "overhead"
        for p in result["fetch"]
        if p["kind"] == "unresolved"
    )


def test_plain_copy_uses_lane_events_without_inventing_a_collective():
    program, diagnostics = fixture()
    program["alias_groups"][0].pop("transfer")
    diagnostics.pop("collectives")
    result = transfer_phase_data(program, diagnostics)
    assert len(result["fetch"]) == 1
    assert result["fetch"][0]["timing"] == "copy-only lane events"
    assert result["summary"][0]["host_copy_gbps"] == pytest.approx(100 / 900)


def test_repeated_aliases_join_to_their_own_parent_and_unmatched_stays_explicit():
    program, diagnostics = fixture()
    first = diagnostics["transfers"]["fetch"]["fetch_0"]
    diagnostics["transfers"]["fetch"]["fetch_1"] = {
        **first,
        "transfer_id": "fetch_1",
        "lane": {
            "lane_started_at_seconds": 1100e-9,
            "lane_finished_at_seconds": 2000e-9,
        },
    }
    records = diagnostics["collectives"]["records"]
    records.append(
        {
            **records[0],
            "started_at_ns": 1400,
            "finished_at_ns": 1900,
            "collective_id": "c1",
        }
    )
    records.append(
        {
            **records[0],
            "started_at_ns": 2400,
            "finished_at_ns": 2900,
            "collective_id": "unmatched",
        }
    )
    result = transfer_phase_data(program, diagnostics)
    linked = [p for p in result["fetch"] if p["kind"] == "collective"]
    assert [(p["transfer_id"], p["record_id"]) for p in linked] == [
        ("fetch_0", "c0"),
        ("fetch_1", "c1"),
    ]
    assert result["unmatched_records"] == 1


def test_missing_collective_time_is_not_used_to_subtract_or_price_copy():
    program, diagnostics = fixture()
    diagnostics["collectives"]["records"][0]["finished_at_ns"] = None
    result = transfer_phase_data(program, diagnostics)
    assert len(result["fetch"]) == 1
    assert result["fetch"][0]["kind"] == "unresolved"
    assert result["summary"] == []


def test_overlapping_collectives_report_sum_and_union_separately():
    program, diagnostics = fixture()
    records = diagnostics["collectives"]["records"]
    records[:] = [
        {**records[0], "source": "task", "started_at_ns": a, "finished_at_ns": b}
        for a, b in [(100, 500), (300, 800)]
    ]
    result = transfer_phase_data(program, diagnostics)
    assert result["summary"][0]["duration_sum_ns"] == 900
    assert result["summary"][0]["busy_ns"] == 700


@pytest.mark.parametrize("field", ["alias_group_id", "transfer_id", "bytes"])
def test_transfer_records_require_payload_identity(field):
    program, diagnostics = fixture()
    diagnostics["transfers"]["fetch"]["fetch_0"].pop(field)
    with pytest.raises(ValueError, match=field):
        transfer_phase_data(program, diagnostics)


@pytest.mark.parametrize("operation,inputs,outputs", [
    ("all_gather", 100, 800),
    ("reduce_scatter", 800, 100),
    ("all_reduce", 800, 800),
])
def test_collective_throughput_counts_one_full_payload(operation, inputs, outputs):
    program, diagnostics = fixture()
    diagnostics["collectives"]["records"][0].update(
        operation=operation, input_bytes=inputs, output_bytes=outputs,
    )
    row = transfer_phase_data(program, diagnostics)["summary"][0]
    assert row["throughput_bytes"] == 800
    assert row["throughput_gbps"] == 1.6


def test_copy_throughput_uses_physical_bytes_and_collectives_use_busy_union():
    program, diagnostics = fixture()
    records = diagnostics["collectives"]["records"]
    records.append({**records[0], "started_at_ns": 500, "finished_at_ns": 1000})
    row = transfer_phase_data(program, diagnostics)["summary"][0]
    assert row["throughput_gbps"] == pytest.approx(1600 / 600)
    program["alias_groups"][0].pop("transfer")
    diagnostics.pop("collectives")
    row = transfer_phase_data(program, diagnostics)["summary"][0]
    assert row["throughput_gbps"] == pytest.approx(100 / 900)
