"""Workers share a backend setup; what names a node's own network path is its own."""

from types import SimpleNamespace

import pytest

from shadowspill.communication.binding import backend as module
from shadowspill.communication.requests import CollectiveRequest
from shadowspill.execution.distributed import worker as distributed_worker


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
    requests = (CollectiveRequest("reduce_scatter", (0, 1), 64, "torch.float32"),)
    rows = [
        record(
            NCCL_SOCKET_IFNAME="enp114s0f1np1", NCCL_DEBUG="WARN", NCCL_HOME="/opt/nccl"
        ),
        record(NCCL_SOCKET_IFNAME="enp4s0f0np0", NCCL_DEBUG="WARN"),
    ]
    setup = module.shared_backend_setup(Group(rows), requests, device_ordinal=None)
    assert setup["environment"] == {"NCCL_PROTO": "simple"}
    assert setup["local_environment"] == [
        {
            "NCCL_SOCKET_IFNAME": "enp114s0f1np1",
            "NCCL_DEBUG": "WARN",
            "NCCL_HOME": "/opt/nccl",
        },
        {"NCCL_SOCKET_IFNAME": "enp4s0f0np0", "NCCL_DEBUG": "WARN"},
    ]
    assert setup["groups"] == {"0,1": {"backend": "nccl"}}
    rows[1]["environment"]["NCCL_PROTO"] = "ll"
    with pytest.raises(ValueError, match=r"backend setups: environment: .*simple.*ll"):
        module.shared_backend_setup(Group(rows), requests, device_ordinal=None)


def test_a_plan_without_collectives_takes_no_backend_setup(monkeypatch):
    backend = {"nccl_version": [2, 29, 7], "groups": {"0,1": {"backend": "nccl"}}}
    requests = []
    monkeypatch.setattr(
        distributed_worker, "collective_requests", lambda program, collectives: requests
    )

    def calibrated(setup):
        return SimpleNamespace(
            program=None,
            communication_info=SimpleNamespace(metadata={"backend_setup": setup}),
        )

    assert not distributed_worker.backend_differs(backend, calibrated({}))
    requests.append(CollectiveRequest("all_gather", (0, 1), 64, "torch.bfloat16"))
    assert distributed_worker.backend_differs(backend, calibrated({}))
    assert not distributed_worker.backend_differs(backend, calibrated(backend))
    assert distributed_worker.backend_differs(
        backend, calibrated({**backend, "nccl_version": [2, 28, 0]})
    )
