"""Worker command and rank-map contracts, independent of any recipe."""

import pytest

from shadowspill.orchestration.launch.processes import worker_command
from shadowspill.orchestration.launch.workers import workers


def test_worker_commands_use_one_process_until_distributed_execution():
    local = worker_command("a.b", ["phase", "request"])
    assert "torch.distributed.run" not in local
    distributed = worker_command("a.b", ["phase", "request"], ranks=2)
    assert "--nproc-per-node=2" in distributed
    assert distributed[-6:] == [
        "-m",
        "shadowspill.orchestration.launch.rank_device",
        "-m",
        "a.b",
        "phase",
        "request",
    ]


def test_worker_mapping_rejects_missing_ranks_before_rendezvous(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with (
        pytest.raises(ValueError, match="every launcher rank"),
        workers(2, rank_map={"0": 1}),
    ):
        pytest.fail("invalid mapping reached a worker")
