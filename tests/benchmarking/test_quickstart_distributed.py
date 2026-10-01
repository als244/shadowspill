from __future__ import annotations

import json
from types import SimpleNamespace

from benchmarking.quickstart.options import _parser, _reproduced_arguments
from benchmarking.quickstart.storage import prepare_run_root
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.pytorch import StepSearchReport


def test_distributed_outputs_and_reproduction_keep_stores_per_rank(
    tmp_path, monkeypatch
):
    parser = _parser()
    arguments = parser.parse_args([])
    arguments.output_dir = tmp_path / "run"
    arguments.distributed = True
    arguments.factory = "example:experiment"
    arguments.build_store = tmp_path / "build"
    request = SimpleNamespace(label="example")
    ranks = [prepare_run_root(arguments, request, rank=rank) for rank in range(2)]
    for rank, paths in enumerate(ranks):
        assert paths.root == tmp_path / "run" / f"rank-{rank:05d}"
        assert paths.store == tmp_path / "run/artifact_store" / f"rank-{rank:05d}"
        assert paths.build_store == tmp_path / "build" / f"rank-{rank:05d}"
        assert paths.plan_store.name == f"rank-{rank:05d}"
        record = json.loads((paths.root / "request.json").read_text())
        assert record["rank"] == rank
        assert record["request"]["artifact_store"] == str(paths.store)
        StepSearchReport(
            metadata={},
            budgets=((2 << 30, 1 << 30),),
            geometries=(),
            points=(),
            transfer_bandwidths=TransferBandwidths(25_000_000_000, 26_000_000_000),
        ).save(paths.root / "search.json")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(
        "sys.argv", ["quickstart", "--reproduce", str(tmp_path / "run")]
    )
    restored = parser.parse_args(["--reproduce", str(tmp_path / "run")])
    restored = _reproduced_arguments(parser, restored)
    assert restored.distributed
    assert restored.artifact_store == tmp_path / "run/artifact_store"
    assert restored.build_store == tmp_path / "build"
    assert restored.plan_store_mode == "require"


def test_text_dp_scaling_is_in_the_recipe_and_preserves_local_metrics(monkeypatch):
    import torch

    from workloads.recipes.text import quickstart

    weight = torch.tensor(2.0, requires_grad=True)
    monkeypatch.setattr(
        quickstart, "full_model_objective", lambda *args: weight.square()
    )
    loss, metrics = quickstart._distributed_objective(None, 4, None)
    loss.backward()
    torch.testing.assert_close(weight.grad, torch.tensor(1.0))
    torch.testing.assert_close(metrics["head_loss"], torch.tensor(4.0))
