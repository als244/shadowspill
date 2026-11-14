"""Generic search, plotting, metric and failure contracts on the phase pipeline."""

import json
import sys

import pytest

from benchmarks.quickstart import main, run
from tests.benchmarking._generic_quickstart import experiment
from tests.shadowspill.offline._example import communication

pytestmark = pytest.mark.cuda


def test_generic_quickstart_search_run_plot(tmp_path):
    from shadowspill import ProfilingOptions, SweepResult

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
            profiling=ProfilingOptions(
                conditioning_seconds=0.001,
                conditioning_max_samples=2,
                measurement_seconds=0.002,
                maximum_samples=4,
            ),
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


def test_chunk_retention_geometry_search_executes_saved_winner_and_replans(tmp_path):
    import torch

    from shadowspill import Program, SweepResult
    from tests.benchmarking._generic_quickstart import chunked_experiment

    output = tmp_path / "chunk_search"
    assert (
        run(
            chunked_experiment,
            search_budget_gib=[2],
            spill_gib=1,
            steps=3,
            output_dir=output,
            plots=False,
            communication_info=communication(),
            warmup_seconds=0.001,
            measure_seconds=0.002,
        )
        == 0
    )
    inventory = json.loads((output / "programs/index.json").read_text())
    assert len(inventory["candidates"]) == 12
    for entry in inventory["candidates"]:
        program = Program.load(output / "programs" / entry["file"])
        sizes = {
            row["row_chunks"]["chunk_size"]
            for row in program.bindings["task_metadata"].values()
            if "row_chunks" in row
        }
        assert sizes == {int(entry["name"].rsplit("chunk", 1)[1])}
    sweep = SweepResult.load(output / "planning/sweep.ssp")
    # Each of three policy pairs gets both chunk sizes and five orderings.
    assert len(sweep.cases) == 30
    assert all(case.plan is not None for case in sweep.cases)
    result = json.loads((output / "execution/budget-0000/result.json").read_text())
    assert result["predicted_seconds"] == pytest.approx(
        min(case.plan.summary["duration_seconds"] for case in sweep.cases)
    )

    setup = chunked_experiment(device="cpu")
    reference = setup["model_factory"]().to_empty(device="cpu")
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            parameter.fill_(0 if name.endswith("bias") else 0.1)
    optimizer = setup["optimizer"](reference.parameters())
    for step in result["steps"]:
        optimizer.zero_grad()
        loss = 0.0
        for args in setup["candidates"][result["candidate"]]["args"]:
            value, _ = setup["objective"](reference, *args)
            value.backward()
            loss += value.item()
        assert step["objective_loss"] == pytest.approx(loss, rel=1e-5)
        optimizer.step()

    artifacts = {
        path: path.read_bytes()
        for directory in ("programs", "profiles", "profiled")
        for path in (output / directory).glob("*.ssp")
    }
    assert len(artifacts) == 36
    assert (
        main(
            [
                "--phase",
                "plan",
                "--output-dir",
                str(output),
                "--search-budget-gib",
                "2.25",
                "--no-plots",
            ]
        )
        == 0
    )
    assert all(path.read_bytes() == value for path, value in artifacts.items())
    assert json.loads((output / "plans/index.json").read_text())[0]["budget"] == int(
        2.25 * (1 << 30)
    )
