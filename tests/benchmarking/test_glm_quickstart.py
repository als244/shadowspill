"""GLM uses the same saved artifacts and fresh execution worker as other models."""

import json

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["glm53_flash", "glm53"])
@pytest.mark.slow
def test_glm_training_from_saved_phases(tmp_path, family):
    from benchmarking.quickstart import run
    from shadowspill.artifacts.sweep import SweepResult
    from tests.shadowspill.offline._example import communication

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert (
        run(
            f"workloads.recipes.text.glm.quickstart:{family}",
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


def test_glm_dp_objective_is_scaled_once_and_data_differs_between_ranks(monkeypatch):
    from types import SimpleNamespace

    from workloads.recipes.text.glm.quickstart import glm53_flash

    def no_live_group(*args, **kwargs):
        pytest.fail("the factory must use logical parallelism, not a live group")

    for name in ("is_initialized", "get_world_size", "get_rank", "init_process_group"):
        monkeypatch.setattr(torch.distributed, name, no_live_group)
    recipes = []
    for rank in (0, 1):
        recipes.append(
            glm53_flash(
                device="cuda:0",
                tiny=True,
                sequence_length=16,
                sequences_per_step=4,
                sequences_per_example=(1, 2),
                ranks=2,
                num_replicas=2,
                logical_rank=rank,
            )
        )
    assert recipes[0]["build_options"]["parallelism"].world_size == 4
    assert recipes[0]["units_per_step"] == 64 * 4
    assert not torch.equal(
        recipes[0]["candidates"]["1"]["args"][0][0],
        recipes[1]["candidates"]["1"]["args"][0][0],
    )
    for recipe in recipes:
        parameter = torch.tensor(2.0, requires_grad=True)
        model = SimpleNamespace(
            loss=lambda *args, value=parameter, **kwargs: value.square()
        )
        loss = recipe["objective"](model, *recipe["candidates"]["1"]["args"][0])
        loss.backward()
        torch.testing.assert_close(parameter.grad, torch.tensor(4.0 / (64 * 4)))
        with recipe["context"]():
            pass
