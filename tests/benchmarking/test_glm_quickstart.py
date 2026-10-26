"""GLM uses the same saved artifacts and fresh execution worker as other models."""

import json

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["glm53_flash", "glm53"])
def test_glm_training_from_saved_phases(tmp_path, family):
    from benchmarking.quickstart import run
    from shadowspill.artifacts.sweep import SweepResult
    from tests.shadowspill.offline._example import communication

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert (
        run(
            f"workloads.models.{family}.quickstart:experiment",
            factory_args={
                "tiny": True,
                "sequence_length": 16,
                "sequences_per_step": 2,
                "sequences_per_example": [1],
            },
            search_budget_gib=[4],
            spill_gib=0.25,
            steps=3,
            output_dir=tmp_path,
            communication_info=communication(),
            warmup_seconds=0.001,
            measure_seconds=0.002,
            external_headroom_gib=0.75,
            plots=False,
        )
        == 0
    )
    sweep = SweepResult.load(tmp_path / "planning/sweep.ssp")
    assert all(case.plan is not None for case in sweep.cases), sweep.cases
    result = json.loads((tmp_path / "execution/budget-0000/result.json").read_text())
    losses = [row["loss"] for row in result["steps"]]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
    assert result["units_per_step"] == 32
    assert result["units_per_second"] > 0
    assert (tmp_path / "report/index.html").is_file()
    assert len(list((tmp_path / "programs").glob("*.ssp"))) == 1
    assert len(list((tmp_path / "profiled").glob("*.ssp"))) == 1
