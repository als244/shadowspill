"""Replanning must not leave old execution results in current comparisons."""

import csv

from benchmarking.quickstart.phases.common import read, write
from benchmarking.quickstart.results import write_measurements


def test_measurements_filter_replaced_plans_and_clear_stale_reports(tmp_path):
    winner = {
        "file": "budget-0000.ssp",
        "candidate": "a",
        "budget": 100,
        "plan_digest": "old",
    }
    value = {
        **winner,
        "predicted_seconds": 0.2,
        "median_seconds": 0.3,
        "units_per_step": 3,
        "unit_label": "samples",
        "units_per_second": 10,
    }
    output = tmp_path / "execution"
    write(output / "budget-0000/result.json", value)
    write_measurements(tmp_path, [winner], plots=False)
    assert read(output / "summary.json")[0]["units_per_second"] == 10
    with (output / "summary.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    for suffix in ("svg", "png"):
        (output / f"performance.{suffix}").write_text("old plot")
    winner["plan_digest"] = "new"
    write_measurements(tmp_path, [winner], plots=False)
    assert read(output / "summary.json") == []
    with (output / "summary.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
    assert not list(output.glob("performance.*"))
