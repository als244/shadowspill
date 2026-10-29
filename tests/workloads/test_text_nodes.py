"""Ranks across machines: the launch helpers and the normal runner's checks."""

from __future__ import annotations

import pytest

from workloads.recipes.text.launch import (
    check_nodes,
    first_rank,
    rank_directories,
    remote_command,
    worker_command,
)

NODES = [
    {"address": "10.0.0.1", "ranks": 2, "env": {"NCCL_SOCKET_IFNAME": "eth1"}},
    {
        "address": "10.0.0.2",
        "ranks": 1,
        "ssh": "peer",
        "repo": "/srv/shadowspill",
        "python": "/srv/env/bin/python",
        "env": {"NCCL_SOCKET_IFNAME": "eth2"},
    },
]


def test_nodes_are_checked_and_ranks_counted():
    assert check_nodes(NODES) == NODES
    assert first_rank(NODES, 1) == 2 and rank_directories(NODES, 1) == ["rank-00002"]
    with pytest.raises(ValueError, match="at least two"):
        check_nodes(NODES[:1])
    with pytest.raises(ValueError, match="launched on"):
        check_nodes([NODES[1], NODES[1]])
    with pytest.raises(ValueError, match="ssh, repo and python"):
        check_nodes([NODES[0], {"address": "x"}])
    from workloads.recipes.text.experiment.inputs import ranks

    assert ranks({"nodes": NODES}) == 3
    with pytest.raises(ValueError, match="added up"):
        ranks({"nodes": NODES, "ranks": 4})


def test_commands_advertise_each_nodes_address(tmp_path):
    assert worker_command("/py", "pkg.build", tmp_path) == [
        "/py",
        "-u",
        "-m",
        "pkg.build",
        str(tmp_path),
    ]
    grouped = worker_command(
        "/py",
        "pkg.execute",
        tmp_path,
        nodes=2,
        node=1,
        ranks=1,
        endpoint="10.0.0.1:29400",
        address="10.0.0.2",
        identity="run-execute",
    )
    assert grouped[:3] == ["/py", "-m", "torch.distributed.run"]
    for flag in (
        "--nnodes=2",
        "--node-rank=1",
        "--rdzv-endpoint=10.0.0.1:29400",
        "--local-addr=10.0.0.2",
    ):
        assert flag in grouped
    assert grouped[-3:] == ["benchmarking.device_worker", "pkg.execute", str(tmp_path)]
    remote = remote_command(NODES[1], grouped, {"A": "b c"})
    assert remote[:4] == ["ssh", "-o", "BatchMode=yes", "peer"]
    assert remote[4].startswith(
        "cd /srv/shadowspill && env A='b c' /py -m torch.distributed.run"
    )
