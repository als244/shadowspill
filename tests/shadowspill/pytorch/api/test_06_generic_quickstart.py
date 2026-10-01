"""Generic quickstart GPU check with no workload/model-library dependencies."""

from __future__ import annotations

import json
import sys

import pytest
import torch
from torch import nn

from benchmarking.quickstart import run
from qualification.profiling import CORRECTNESS_PROFILING

pytestmark = [pytest.mark.fresh_process, pytest.mark.cuda]


def experiment(*, device):
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(8, 4, generator=generator)
    target = torch.randn(8, 2, generator=generator)

    def model_factory():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))

    def objective(model, features, target):
        loss = (model(features) - target).square().sum() / 16
        return loss, {"residual": loss.detach()}

    return {
        "model_factory": model_factory,
        "objective": objective,
        "optimizer": lambda params: torch.optim.SGD(params, lr=0.02),
        "hyperparams": {"lr": 0.02},
        "candidates": {
            name: tuple(
                (features[i : i + n].clone(), target[i : i + n].clone())
                for i in range(0, 8, n)
            )
            for name, n in [("two_images", 2), ("four_images", 4)]
        },
        "units_per_step": 8,
        "unit_label": "images",
    }


def test_generic_quickstart_search_run_plot(tmp_path):
    output = tmp_path / "quickstart"
    stdout = sys.stdout
    assert (
        run(
            experiment,
            search_budget_gib=[2, 2.25],
            spill_gib=1,
            steps=3,
            output_dir=output,
            device="cuda:0",
            plots=True,
            timelines=True,
            resolution_plans=True,
            profiling_options=CORRECTNESS_PROFILING,
        )
        == 0
    )
    report = json.loads((output / "search.json").read_text())
    assert report["metadata"]["unit_label"] == "images"
    assert {point["candidate"] for point in report["points"]} == {
        "two_images",
        "four_images",
    }
    assert len(tuple((output / "steps").glob("*.json"))) == 2
    assert (output / "figures" / "real" / "throughput.png").stat().st_size > 0
    assert (output / "figures" / "raw_data" / "run_budgets.csv").is_file()

    assert sys.stdout is stdout
    assert (output / "timelines" / "index.html").is_file()
    assert len((output / "step_metrics.jsonl").read_text().splitlines()) == 6
