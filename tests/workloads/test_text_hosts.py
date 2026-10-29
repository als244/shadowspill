"""Several hosts share one run: the config checks, the commands and the layout."""

from __future__ import annotations

import json

import pytest

from workloads.recipes.text.experiment import targets
from workloads.recipes.text.experiment.hosts import (
    check_hosts,
    remote_command,
    worker_command,
)

HOSTS = [
    {"address": "10.0.0.1", "ranks": 1, "env": {"NCCL_SOCKET_IFNAME": "eth1"}},
    {
        "address": "10.0.0.2",
        "ranks": 1,
        "ssh": "peer",
        "repo": "/srv/shadowspill",
        "python": "/srv/env/bin/python",
        "env": {"NCCL_SOCKET_IFNAME": "eth2"},
    },
]


def test_hosts_are_checked():
    raw = {"hosts": HOSTS, "max_tokens_per_microbatch": 2048}
    assert check_hosts(raw) == HOSTS
    assert check_hosts({}) is None
    with pytest.raises(ValueError, match="at least two"):
        check_hosts({"hosts": HOSTS[:1], "max_tokens_per_microbatch": 2048})
    with pytest.raises(ValueError, match="launched on"):
        check_hosts({"hosts": [HOSTS[1], HOSTS[1]], "max_tokens_per_microbatch": 2048})
    with pytest.raises(ValueError, match="ssh, repo and python"):
        check_hosts(
            {"hosts": [HOSTS[0], {"address": "x"}], "max_tokens_per_microbatch": 2048}
        )
    with pytest.raises(ValueError, match="added up"):
        check_hosts({**raw, "ranks": 3})
    with pytest.raises(ValueError, match="fixed max_tokens_per_microbatch"):
        check_hosts({"hosts": HOSTS})


def test_commands_name_the_rendezvous_and_the_remote_repository(tmp_path):
    single = worker_command("/py", "pkg.build", tmp_path)
    assert single == ["/py", "-u", "-m", "pkg.build", str(tmp_path)]
    grouped = worker_command(
        "/py",
        "pkg.execute",
        tmp_path,
        nodes=2,
        node=1,
        ranks=1,
        endpoint="10.0.0.1:29400",
        identity="run",
    )
    assert grouped[:3] == ["/py", "-m", "torch.distributed.run"]
    assert "--nnodes=2" in grouped and "--node-rank=1" in grouped
    assert "--rdzv-endpoint=10.0.0.1:29400" in grouped
    assert grouped[-3:] == ["benchmarking.device_worker", "pkg.execute", str(tmp_path)]
    remote = remote_command(
        HOSTS[1], grouped, {"SHADOWSPILL_HOST_INDEX": "1", "A": "b c"}
    )
    assert remote[:4] == ["ssh", "-o", "BatchMode=yes", "peer"]
    assert remote[4].startswith(
        "cd /srv/shadowspill && env SHADOWSPILL_HOST_INDEX=1 A='b c' /py -m"
    )


def test_targets_live_under_the_run_for_one_host_and_per_gpu_for_several(tmp_path):
    assert targets.artifacts(tmp_path, {}) == tmp_path
    assert targets.read(tmp_path, {}) == {"tag": None, "host": 0}
    with pytest.raises(FileNotFoundError, match="names this host's target"):
        targets.artifacts(tmp_path, {"hosts": HOSTS})
    (tmp_path / "target.json").write_text(json.dumps({"tag": "card_sm90", "host": 1}))
    assert (
        targets.artifacts(tmp_path, {"hosts": HOSTS})
        == tmp_path / "targets" / "card_sm90"
    )
    assert targets.first_rank({"hosts": [{"ranks": 2}, {"ranks": 1}]}, 1) == 2
    assert targets.first_rank({"ranks": 4}, 0) == 0
