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


def test_quickstart_releases_resources_after_imported_state(monkeypatch):
    from benchmarking.quickstart import runner

    events = []
    tour = object.__new__(runner.Tour)
    tour.model, tour.runtime = object(), object()
    tour.experiment = {
        "cleanup_model": lambda model: events.append(("resources", model))
    }
    monkeypatch.setattr(
        runner,
        "release_model_state",
        lambda model, **kw: events.append(("state", model)),
    )
    tour._release_model()
    assert events == [("state", tour.model), ("resources", tour.model)]


def test_quickstart_threads_are_configurable_per_rank():
    from benchmarking.quickstart.options import _parser, search_policy

    args = _parser().parse_args(["--search-workers", "4"])
    assert search_policy(args).workers == 4


def test_symmetric_cli_override_preserves_the_callers_specification(monkeypatch):
    from benchmarking.quickstart import runner
    from shadowspill.pytorch import Distributed

    parser = _parser()
    assert parser.parse_args([]).symmetric_planning is None
    assert parser.parse_args(["--no-symmetric-planning"]).symmetric_planning is False
    arguments = parser.parse_args(["--symmetric-planning"])
    specification = Distributed(None)
    model, received = object(), []
    tour = object.__new__(runner.Tour)
    tour.arguments, tour.runtime, tour.distributed = arguments, object(), specification
    tour.experiment = {"model_factory": lambda: model}
    monkeypatch.setattr(runner, "initialize_model", lambda value, **_: value)
    monkeypatch.setattr(
        runner,
        "import_model_state",
        lambda value, **kw: received.append(kw["distributed"]),
    )
    tour._build_model()
    assert received[0].symmetric_planning is True
    assert specification.symmetric_planning is False
