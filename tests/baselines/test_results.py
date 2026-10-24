"""The row an engine writes and the measurement contract."""

import json

import pytest

from baselines.common import results
from baselines.common.results import (
    RESULT_SCHEMA,
    host_peak_bytes,
    memory_peaks,
    read_rows,
    row,
    summarize_cycles,
    validate_row,
    write_json,
)


def test_slowest_rank_is_selected_per_step_before_taking_median():
    report = summarize_cycles([[1, 10, 1], [10, 1, 10]], global_tokens=100)
    assert report["step_seconds"] == 10
    assert report["global_tokens_per_second"] == 10
    assert report["slowest_rank_cycle_seconds"] == [10, 10, 10]
    assert report["useful_tflops_per_gpu"] is None


def test_useful_work_is_credited_per_device():
    report = summarize_cycles(
        [[2.0, 2.0], [2.0, 2.0]], global_tokens=100, useful_flops_per_step=8e12
    )
    assert report["useful_tflops_per_gpu"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "cycles", [[], [[]], [[1, 2], [1]], [[1, 0]], [[1, float("nan")]], [[-1.0]]]
)
def test_partial_or_invalid_measurements_cannot_be_reported(cycles):
    with pytest.raises(ValueError):
        summarize_cycles(cycles, global_tokens=100)


def test_the_measurement_keys_are_what_summarize_cycles_reports():
    report = summarize_cycles([[1.0]], global_tokens=1)
    assert set(results.MEASUREMENT_KEYS) == set(report)


def test_write_json_creates_parents_and_leaves_no_temporary(tmp_path):
    path = tmp_path / "deep" / "row.json"
    write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert list(path.parent.iterdir()) == [path]


def peaks(allocated, reserved, host):
    return {
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "host_peak_rss_bytes": host,
    }


def completed_row(**changes):
    fields = {
        "engine": "fsdp2",
        "status": "completed",
        "model": {"model": "llama3-tiny", "implementation": "pytorch", "mode": "full"},
        "geometry": {"sequence_length": 4096},
        "precision": {"model_dtype": "bfloat16"},
        "configuration": {"recompute": 1.0},
        "hardware": {"host": "chicago"},
        "provenance": {"git_head": "abc"},
        "measurement": summarize_cycles([[1.0, 1.0]], global_tokens=4096),
        "memory": memory_peaks([peaks(1, 2, 3)]),
    }
    fields.update(changes)
    return row(**fields)


def test_a_completed_row_carries_its_schema_measurement_and_peaks():
    value = completed_row()
    assert value["schema"] == RESULT_SCHEMA
    assert value["failure"] is None
    validate_row(value)
    with pytest.raises(ValueError, match="measurement"):
        completed_row(measurement={"step_seconds": 1.0})
    with pytest.raises(ValueError, match="memory"):
        completed_row(memory={"peak_allocated_bytes": 1})


def test_an_out_of_memory_row_is_a_row_and_a_failure_names_its_cause():
    boundary = completed_row(status="cuda_out_of_memory", measurement=None, memory=None)
    assert boundary["measurement"] is None
    with pytest.raises(ValueError):
        completed_row(status="failed", measurement=None, memory=None)
    failed = completed_row(
        status="failed", measurement=None, memory=None, failure={"error": "x"}
    )
    assert failed["failure"] == {"error": "x"}
    with pytest.raises(ValueError):
        completed_row(status="exploded")
    with pytest.raises(ValueError):
        completed_row(model={"implementation": "pytorch"})
    with pytest.raises(ValueError):
        validate_row({**completed_row(), "schema": "other/v0"})


def test_device_peaks_are_the_binding_rank_and_host_peaks_are_summed():
    combined = memory_peaks([peaks(10, 12, 5), peaks(7, 20, 6)])
    assert combined["peak_allocated_bytes"] == 10
    assert combined["peak_reserved_bytes"] == 20
    assert combined["host_peak_rss_bytes"] == 11
    assert "sum" in combined["host_peak_rss_aggregation"]
    assert combined["host_peak_bytes"] == 11


def test_shadowspill_host_peak_uses_all_rank_plans_and_preserves_rss():
    value = completed_row(engine="shadowspill")
    value["memory"]["shadowspill"] = [
        {"predicted_spill_peak_bytes": 10},
        {"predicted_spill_peak_bytes": 20},
    ]
    assert host_peak_bytes(value) == 30
    assert value["memory"]["host_peak_rss_bytes"] == 3
    value["memory"]["shadowspill"][1] = {}
    assert host_peak_bytes(value) == 3


def test_host_peak_accepts_rows_written_before_the_common_field():
    value = completed_row()
    del value["memory"]["host_peak_bytes"]
    assert host_peak_bytes(value) == 3


def test_read_rows_keeps_the_last_attempt_of_a_case(tmp_path):
    case = tmp_path / "cases" / "case-0001-abc"
    first = completed_row(
        status="failed", measurement=None, memory=None, failure={"error": "x"}
    )
    write_json(case / "attempt-001" / "row.json", first)
    write_json(case / "attempt-002" / "row.json", completed_row())
    write_json(tmp_path / "standalone" / "row.json", completed_row())
    assert [value["status"] for value in read_rows(tmp_path)] == [
        "completed",
        "completed",
    ]
    every = read_rows(tmp_path, latest_attempts=False)
    assert [value["status"] for value in every] == ["failed", "completed", "completed"]


def test_read_rows_collects_every_row_under_a_root_and_refuses_a_bad_one(tmp_path):
    first = completed_row()
    second = completed_row(status="cuda_out_of_memory", measurement=None, memory=None)
    write_json(tmp_path / "round1" / "fsdp2" / "case-0001" / "row.json", first)
    write_json(tmp_path / "round1" / "fsdp2" / "case-0002" / "row.json", second)
    write_json(tmp_path / "round1" / "fsdp2" / "case-0002" / "request.json", {"x": 1})
    rows = read_rows(tmp_path)
    assert [value["status"] for value in rows] == ["completed", "cuda_out_of_memory"]
    assert rows[0]["_path"].endswith("case-0001/row.json")
    bad = tmp_path / "round1" / "fsdp2" / "case-0003" / "row.json"
    write_json(bad, {"schema": "?"})
    with pytest.raises(ValueError):
        read_rows(tmp_path)
