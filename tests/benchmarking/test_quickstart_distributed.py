from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarking.quickstart.options import _parser, _reproduced_arguments
from benchmarking.quickstart.storage import prepare_run_root
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.pytorch import StepSearchReport


def sharded_factory(*, device, transfer_shard_size=None, shard_gradients=False):
    return (device, transfer_shard_size, shard_gradients)


@pytest.mark.parametrize("local_fixed", [1 << 30, (1 << 30) + (32 << 20)])
def test_distributed_budgets_use_the_smallest_available_slab(monkeypatch, local_fixed):
    from benchmarking.quickstart.runner import plan_budgets

    group = object()
    runtime = SimpleNamespace(
        control_group=group,
        pools={
            "execution": SimpleNamespace(
                physical_capacity=70 << 30, capacity=(70 << 30) - local_fixed
            )
        },
    )
    request = SimpleNamespace(
        search_budgets=[16 << 30, 70 << 30], run_budgets=[70 << 30]
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _: 2)

    def exchange(peers, value, *, group):
        requested, _ = value
        peers[:] = [(requested, 1 << 30), (requested, (1 << 30) + (32 << 20))]

    monkeypatch.setattr(torch.distributed, "all_gather_object", exchange)
    budgets = plan_budgets(runtime, request)
    assert budgets.search == [(15 << 30) - (32 << 20), (69 << 30) - (32 << 20)]
    assert budgets.run == budgets.search[-1:]
    assert all(value <= runtime.pools["execution"].capacity for value in budgets.search)


def test_distributed_budget_mismatch_is_rejected_before_search(monkeypatch):
    from benchmarking.quickstart.runner import plan_budgets

    runtime = SimpleNamespace(
        control_group=object(),
        pools={
            "execution": SimpleNamespace(physical_capacity=8 << 30, capacity=7 << 30)
        },
    )
    request = SimpleNamespace(search_budgets=[8 << 30], run_budgets=[8 << 30])
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _: 2)

    def exchange(peers, value, *, group):
        peers[:] = [value, (((7 << 30,), (7 << 30,)), 1 << 30)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", exchange)
    with pytest.raises(ValueError, match="requested different budgets"):
        plan_budgets(runtime, request)


def test_sharding_flags_reach_supporting_factories(monkeypatch):
    from benchmarking.quickstart import cli
    from benchmarking.quickstart.options import parse_arguments

    monkeypatch.setattr(cli, "resolve_device", lambda _: torch.device("cuda:0"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "quickstart",
            "--factory",
            f"{__name__}:sharded_factory",
            "--distributed",
            "--transfer-shard-size",
            "4",
            "--shard-gradients",
        ],
    )
    parser, arguments = parse_arguments()
    _, setup = cli.resolve_request(parser, arguments)
    assert setup(device="cuda:0") == ("cuda:0", 4, True)
    arguments.factory_args = {"transfer_shard_size": 2}
    with pytest.raises(SystemExit):
        cli.resolve_request(parser, arguments)


def test_glm_dp_objective_is_scaled_once_and_data_differs_between_ranks(monkeypatch):
    from workloads.models.glm53_flash import quickstart as glm

    def no_live_group(*args, **kwargs):
        pytest.fail("the factory must use logical parallelism, not a live group")

    for name in ("is_initialized", "get_world_size", "get_rank", "init_process_group"):
        monkeypatch.setattr(torch.distributed, name, no_live_group)
    recipes = []
    for rank in (0, 1):
        recipes.append(
            glm.experiment(
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
            transfer_bandwidths=TransferBandwidths(
                25_000_000_000, 25_000_000_000, 26_000_000_000, 26_000_000_000
            ),
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
    model, received = torch.nn.Module(), []
    tour = object.__new__(runner.Tour)
    tour.arguments, tour.runtime, tour.distributed = arguments, object(), specification
    tour.experiment = {"model_factory": lambda: model}
    # The build reads the request for a pool image; this run has none.
    tour.request = runner.Request(
        label="t",
        search_budgets=[1 << 30],
        run_budgets=[1 << 30],
        physical_capacity=1 << 30,
        spill_budget=1 << 30,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(runner, "initialize_model", lambda value, **_: value)
    monkeypatch.setattr(
        runner,
        "import_model_state",
        lambda value, **kw: received.append(kw["distributed"]),
    )
    tour._build_model()
    assert received[0].symmetric_planning is True
    assert specification.symmetric_planning is False
