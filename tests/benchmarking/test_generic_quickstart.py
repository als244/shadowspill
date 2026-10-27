"""Generic search, plotting, metric and failure contracts on the phase pipeline."""

import json
import sys

import pytest

from benchmarking.quickstart import main, run
from tests.benchmarking._generic_quickstart import experiment
from tests.shadowspill.offline._example import communication

pytestmark = [pytest.mark.fresh_process, pytest.mark.cuda]


def test_generic_quickstart_search_run_plot(tmp_path):
    from shadowspill import SweepResult

    output = tmp_path / "quickstart"
    stdout = sys.stdout
    assert (
        run(
            experiment,
            search_budget_gib=[2, 2.25],
            spill_gib=1,
            steps=3,
            output_dir=output,
            plots=True,
            communication_info=communication(),
            warmup_seconds=0.001,
            measure_seconds=0.002,
        )
        == 0
    )
    assert sys.stdout is stdout
    report = SweepResult.load(output / "planning/sweep.ssp")
    assert {case.candidate for case in report.cases} == {
        "two_images: 1x4rp",
        "two_images: 2x2rp",
        "two_images: 4x1rp",
        "four_images: 1x2rp",
        "four_images: 2x1rp",
    }
    assert len(report.cases) == 10  # Five orderings at each of two budgets.
    assert (output / "report/index.html").is_file()
    assert list((output / "report").rglob("timeline.svg"))
    assert (output / "execution/performance.png").stat().st_size > 0
    assert (output / "execution/summary.csv").is_file()
    runs = list((output / "execution").glob("budget-*/result.json"))
    assert len(runs) == 2
    for path in runs:
        result = json.loads(path.read_text())
        assert result["unit_label"] == "images" and result["units_per_step"] == 8
        assert len(result["steps"]) == 3
        metrics = output / result["run_directory"] / "metrics.jsonl"
        records = [json.loads(line) for line in metrics.read_text().splitlines()]
        assert len(records) == 3
        for row, step in zip(records, result["steps"], strict=True):
            assert isinstance(row["train/residual"], float)
            assert row["train/loss"] == step["loss"]
            assert row["train/residual"] == pytest.approx(step["objective_loss"])
            assert 2 * step["loss"] == pytest.approx(step["objective_loss"])


def test_generic_quickstart_reducer_failure_releases_outputs(tmp_path):
    output = tmp_path / "failed_quickstart"
    with pytest.raises(RuntimeError, match="quickstart execute failed"):
        run(
            experiment,
            factory_args={"fail_on_step": 2},
            search_budget_gib=[2],
            spill_gib=1,
            steps=3,
            output_dir=output,
            plots=False,
            communication_info=communication(),
            warmup_seconds=0.001,
            measure_seconds=0.002,
        )
    log = (output / "logs/execute-0000.log").read_text()
    assert "intentional reducer failure at step 2" in log
    assert "cleanup also failed" not in log
    assert not list((output / "execution").glob("*/result.json"))
    metrics = next((output / "execution").rglob("metrics.jsonl"))
    assert len(metrics.read_text().splitlines()) == 1
    # Retry execution only: the same saved Plan remains usable after the failed
    # worker exits, without rebuilding or measuring a new model.
    assert (
        main(
            [
                "--phase",
                "execute",
                "--output-dir",
                str(output),
                "--factory-args",
                '{"fail_on_step":0}',
            ]
        )
        == 0
    )
    result = json.loads(next((output / "execution").glob("*/result.json")).read_text())
    assert len(result["steps"]) == 3
