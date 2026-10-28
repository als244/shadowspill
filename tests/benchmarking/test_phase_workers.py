"""Phase launching and result aggregation stay independent of the model."""

import pytest

from benchmarking.processes import worker_command
from benchmarking.quickstart.phases.common import write
from benchmarking.quickstart.phases.distributed import preparation_size, workers
from benchmarking.quickstart.phases.execute import aggregate_steps


def test_worker_commands_use_one_process_until_distributed_execution():
    local = worker_command("a.b", ["phase", "request"])
    assert "torch.distributed.run" not in local
    distributed = worker_command("a.b", ["phase", "request"], ranks=2)
    assert "--nproc-per-node=2" in distributed
    assert distributed[-5:] == [
        "-m",
        "benchmarking.device_worker",
        "a.b",
        "phase",
        "request",
    ]


def test_only_communicating_resources_require_distributed_preparation():
    from shadowspill import Parallelism, Resource

    parallelism = Parallelism.dp(ranks_per_replica=4, num_replicas=2)
    setup = {"build_options": {"parallelism": parallelism}}
    assert preparation_size(setup) == 1
    setup["build_options"]["resources"] = {"local": Resource("backend:initialize")}
    assert preparation_size(setup) == 1
    setup["build_options"]["resources"]["collective"] = Resource(
        "backend:initialize", groups={"group": parallelism.group("all_ranks")}
    )
    assert preparation_size(setup) == 8


def test_execution_uses_slowest_rank_and_retains_global_loss():
    ranks = [
        {
            "steps": [
                {
                    "step": 1,
                    "seconds": seconds,
                    "loss": loss,
                    "objective_loss": loss + 0.1,
                }
            ]
        }
        for seconds, loss in ((0.5, 1), (0.7, 2))
    ]
    assert aggregate_steps(ranks) == [
        {"step": 1, "seconds": 0.7, "loss": 3, "objective_loss": 3.2}
    ]
    ranks[1]["steps"][0]["step"] = 2
    with pytest.raises(ValueError, match="different training steps"):
        aggregate_steps(ranks)


def test_worker_mapping_rejects_missing_ranks_before_rendezvous(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with (
        pytest.raises(ValueError, match="every launcher rank"),
        workers({"rank_map": {"0": 1}}, {"world_size": 2}),
    ):
        pytest.fail("invalid mapping reached a worker")


def test_communication_can_use_supplied_artifact_without_torch_or_model(tmp_path):
    import os
    import subprocess
    import sys

    from tests.shadowspill.offline._example import communication

    source = tmp_path / "supplied.ssp"
    communication().save(source)
    write(tmp_path / "programs/index.json", {"world_size": 8})
    write(
        tmp_path / "request.json",
        {"output_dir": str(tmp_path), "communication_info": str(source)},
    )
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from benchmarking.quickstart.phases.worker import main
sys.argv = ['worker', 'communicate', sys.argv[1]]
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            script,
            str(tmp_path / "request.json"),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "communication.ssp").is_file()


@pytest.mark.cuda
def test_distributed_quickstart_profiles_locally_then_launches_mapped_workers(tmp_path):
    import json

    import torch

    from benchmarking.quickstart import run
    from shadowspill import CommunicationInfo

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct CUDA devices")
    assert (
        run(
            "workloads.recipes.regression.benchmark:experiment",
            factory_args={"rows": 8, "width": 8, "outputs": 8, "ranks": 2},
            rank_map={0: 1, 1: 0},
            search_budget_gib=[3],
            spill_gib=0.125,
            steps=3,
            output_dir=tmp_path,
            external_headroom_gib=0.75,
            warmup_seconds=0.001,
            measure_seconds=0.002,
            plots=False,
        )
        == 0
    )
    communication = CommunicationInfo.load(tmp_path / "communication.ssp")
    assert {item["operation"] for item in communication.measurements} >= {
        "all_gather",
        "reduce_scatter",
    }
    result = json.loads((tmp_path / "execution/budget-0000/result.json").read_text())
    assert result["units_per_step"] == 16
    assert [rank["logical_rank"] for rank in result["per_rank"]] == [0, 1]
    assert result["steps"][-1]["loss"] < result["steps"][0]["loss"]
    for index, step in enumerate(result["steps"]):
        ranks = [rank["steps"][index] for rank in result["per_rank"]]
        assert step["seconds"] == max(rank["seconds"] for rank in ranks)
        assert step["loss"] == sum(rank["loss"] for rank in ranks)
    assert len(list((tmp_path / "execution").glob("*/run-*/rank-*/metrics.jsonl"))) == 2
