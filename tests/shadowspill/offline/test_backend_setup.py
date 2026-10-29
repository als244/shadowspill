"""Workers share a backend setup; what names a node's own network path is its own."""

import pytest

from shadowspill.communication.binding import backend as module
from shadowspill.communication.requests import CollectiveRequest


class Group:
    logical_rank = 0

    def __init__(self, rows):
        self.rows = rows

    def for_members(self, ranks):
        return ranks

    def check(self, failure):
        if failure:
            raise ValueError(failure)

    def gather(self, local):
        return self.rows


def record(**environment):
    return {
        "groups": {"0,1": {"backend": "nccl"}},
        "nccl_version": [2, 29, 7],
        "environment": {"NCCL_PROTO": "simple", **environment},
        "config_files": {},
    }


def test_each_nodes_network_settings_are_its_own(monkeypatch):
    monkeypatch.setattr(
        module, "backend_setup", lambda groups, device_ordinal=None: record()
    )
    requests = (CollectiveRequest("reduce_scatter", (0, 1), 64),)
    rows = [
        record(NCCL_SOCKET_IFNAME="enp114s0f1np1", NCCL_DEBUG="WARN"),
        record(NCCL_SOCKET_IFNAME="enp4s0f0np0", NCCL_DEBUG="WARN"),
    ]
    setup = module.shared_backend_setup(Group(rows), requests, device_ordinal=None)
    assert setup["environment"] == {"NCCL_PROTO": "simple"}
    assert setup["local_environment"] == [
        {"NCCL_SOCKET_IFNAME": "enp114s0f1np1", "NCCL_DEBUG": "WARN"},
        {"NCCL_SOCKET_IFNAME": "enp4s0f0np0", "NCCL_DEBUG": "WARN"},
    ]
    assert setup["groups"] == {"0,1": {"backend": "nccl"}}
    rows[1]["environment"]["NCCL_PROTO"] = "ll"
    with pytest.raises(ValueError, match="different communication backend setups"):
        module.shared_backend_setup(Group(rows), requests, device_ordinal=None)
