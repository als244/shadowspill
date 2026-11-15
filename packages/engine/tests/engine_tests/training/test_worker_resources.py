"""Host capacity includes replicas, overhead and independent artifact arenas."""

from dataclasses import replace

import pytest

from engine_tests.training.test_worker_startup import prepared_worker
from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ResourcesUnavailable,
)
from shadowspill.orchestration.launch.host_arena import HostArena
from shadowspill.training.worker.resources import WorkerHost, worker_request
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES


def startup(rank):
    original = prepared_worker().startup
    return replace(
        original,
        scope=replace(original.scope, rank=rank, worker_incarnation=f"worker-{rank}"),
        logical_by_worker=(1, 0),
        device_bytes=100,
        spill_bytes=200,
        staging_bytes=300,
        diagnostics_bytes=400,
    )


def test_two_physical_replicas_add_all_host_demands_with_one_serial_staging_each():
    one, two = startup(0), startup(1)
    allowance = WorkerHost(1000, 2000)
    expected = 200 + 300 + 400 + 1000 + 2000 + 8 * MAX_METADATA_BYTES + (128 << 10)
    assert allowance.peak_bytes(one) == expected
    request = worker_request({"gpu-a": (one, allowance), "gpu-b": (two, allowance)})
    assert dict(request.devices) == {"gpu-a": 100, "gpu-b": 100}
    assert request.host_bytes == 2 * expected
    node = NodeAdmission(
        NodeCapacity(
            "local",
            (AssignedDevice("gpu-a", "0", 100), AssignedDevice("gpu-b", "1", 100)),
            2 * expected + 100,
        )
    )
    with HostArena(node, 101, 1):
        with pytest.raises(ResourcesUnavailable):
            node.reserve(request)
        assert not node.used.devices  # The rejected group retains no partial GPU set.
    with node.reserve(request):
        assert node.used.host_bytes == 2 * expected
    assert node.used.claims == 0


@pytest.mark.parametrize("fault", ["rank", "run", "incarnation", "bundle", "mapping"])
def test_a_node_worker_set_must_describe_one_consistent_live_run(fault):
    one, two = startup(0), startup(1)
    if fault == "rank":
        two = replace(two, scope=one.scope)
    elif fault == "run":
        two = replace(two, scope=replace(two.scope, run_id="other"))
    elif fault == "incarnation":
        two = replace(two, scope=replace(two.scope, server_incarnation="other"))
    elif fault == "bundle":
        two = replace(two, bundle=replace(two.bundle, id="other"))
    else:
        two = replace(two, logical_by_worker=(0, 1))
    with pytest.raises(ValueError, match="disagree"):
        worker_request({"a": (one, WorkerHost(1, 1)), "b": (two, WorkerHost(1, 1))})


def test_process_and_decoded_artifact_allowances_cannot_be_omitted_or_zeroed():
    for values in ((0, 1), (1, 0), (True, 1), (1, -1)):
        with pytest.raises((TypeError, ValueError)):
            WorkerHost(*values)
    with pytest.raises(ValueError, match="nonempty"):
        worker_request({})
