from dataclasses import replace

import pytest

from shadowspill import CommunicationInfo
from shadowspill.communication.collectives import collective_cost


def info():
    return CommunicationInfo(
        fetch_bytes_per_second=25e9,
        evict_bytes_per_second=25e9,
        scale_up_bytes_per_second=300e9,
        scale_up_latency_seconds=2e-6,
        topology={"node0": tuple(range(8))},
    )


def cost(communication, operation="all_gather", ranks=tuple(range(8)), **kwargs):
    return collective_cost(
        communication,
        operation=operation,
        ranks=ranks,
        tensor_bytes=1 << 30,
        dtype="torch.bfloat16",
        setup=kwargs.pop("setup", {"gpu": "fixture"}),
        **kwargs,
    )


def test_gather_scatter_use_total_tensor_not_host_shard_bytes():
    expected_bytes = 7 * (1 << 27)
    for operation in ("all_gather", "reduce_scatter"):
        actual = cost(info(), operation)
        assert actual.transfer_bytes == expected_bytes
        assert actual.duration_seconds == pytest.approx(
            expected_bytes / 300e9 + 7 * 2e-6
        )
        assert actual.source == "ring"
    reduced = cost(info(), "all_reduce")
    assert reduced.transfer_bytes == expected_bytes * 2
    assert reduced.duration_seconds == pytest.approx(actual.duration_seconds * 2)


def test_measurement_requires_exact_group_geometry_dtype_and_setup(tmp_path):
    measurement = dict(
        operation="all_gather",
        ranks=tuple(range(8)),
        tensor_bytes=1 << 30,
        dtype="torch.bfloat16",
        setup={"gpu": "fixture"},
        topology=info().topology,
        duration_seconds=0.001,
    )
    calibrated = replace(info(), measurements=(measurement,))
    calibrated.save(tmp_path / "communication.ssp")
    loaded = CommunicationInfo.load(tmp_path / "communication.ssp")
    assert cost(loaded).source == "measurement"
    assert cost(loaded).duration_seconds == 0.001
    assert cost(loaded, setup={"gpu": "different"}).source == "ring"
    assert cost(loaded, ranks=tuple(reversed(range(8)))).source == "ring"
    assert cost(loaded, "reduce_scatter").source == "ring"
    moved = replace(
        loaded,
        topology={"node0": tuple(range(4)), "node1": tuple(range(4, 8))},
        scale_out_bytes_per_second=25e9,
        scale_out_latency_seconds=10e-6,
    )
    assert cost(moved).source == "ring"
    newer = {**measurement, "duration_seconds": 0.002}
    assert (
        cost(replace(loaded, measurements=(measurement, newer))).duration_seconds
        == 0.002
    )


def test_inter_node_ring_uses_bottleneck_link_without_double_counting_host():
    two_nodes = replace(
        info(),
        topology={"node0": (0, 1), "node1": (2, 3)},
        scale_out_bytes_per_second=25e9,
        scale_out_latency_seconds=10e-6,
    )
    result = cost(two_nodes, ranks=(0, 1, 2, 3))
    assert result.duration_seconds == pytest.approx(3 * (10e-6 + (1 << 28) / 25e9))
    assert (
        cost(replace(two_nodes, fetch_bytes_per_second=1), ranks=(0, 1, 2, 3)) == result
    )


def test_missing_costs_fail_and_identity_needs_no_links():
    assert cost(info(), ranks=(0,)).duration_seconds == 0
    with pytest.raises(ValueError, match="requires topology"):
        cost(replace(info(), topology={}))
    with pytest.raises(ValueError, match="scale_up rate and latency"):
        cost(replace(info(), scale_up_latency_seconds=None))
    with pytest.raises(ValueError, match="scale_out rate and latency"):
        cost(replace(info(), topology={"node0": (0, 1, 2, 3), "node1": (4, 5, 6, 7)}))
