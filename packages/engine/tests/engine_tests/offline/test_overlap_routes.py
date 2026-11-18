"""Route identity, traffic coordinates and profile overlays stay CPU-only."""

import math
from dataclasses import replace
from itertools import pairwise

from shadowspill import CommunicationInfo, Profiles, TaskProfile, apply_profiles
from shadowspill.communication.overlap import (
    MAX_ROUTE_SIZES,
    overlap_routes,
    spread_sizes,
    timing_class,
)
from shadowspill.communication.requests import CollectiveRequest
from shadowspill.ir import TaskOverlapCost

from ._example import profiles, program


def test_route_identity_tracks_configuration_but_not_message_size():
    request = CollectiveRequest("all_gather", (0, 1), 1024, "torch.float32", True)
    info = CommunicationInfo(1e9, 1e9, topology={"node0": (0, 1)})
    identity = timing_class((request,), info)
    assert timing_class((replace(request, tensor_bytes=2048),), info) == identity
    for changed in (
        replace(request, operation="reduce_scatter"),
        replace(request, dtype="torch.bfloat16"),
        replace(request, ranks=(1, 0)),
        replace(request, inplace=False),
    ):
        assert timing_class((changed,), info) != identity
    assert (
        timing_class((request,), replace(info, topology={"a": (0,), "b": (1,)}))
        != identity
    )
    assert (
        timing_class(
            (request,),
            replace(info, metadata={"backend_setup": {"nccl_version": [2, 30]}}),
        )
        != identity
    )


def test_overlap_overlay_preserves_measured_timing_memory_and_round_trips(tmp_path):
    source = profiles()
    source.tasks["update_impl"] = replace(
        source.tasks["update_impl"],
        metadata={"duration_kind": "measured", "memory_kind": "measured"},
    )
    baseline = source.tasks["update_impl"]
    cost = TaskOverlapCost("collective_test", 128, 0.75, 0.8)
    overlay = Profiles(
        setup=source.setup,
        tasks={
            "update_impl": TaskProfile(
                overlap_costs=(cost,),
                metadata={"overlap_calibration": {"status": "measured"}},
            )
        },
    )
    overlay.save(tmp_path / "overlap.ssp")
    merged = source.merge(Profiles.load(tmp_path / "overlap.ssp"))
    record = merged.tasks["update_impl"]
    assert record.duration_seconds == baseline.duration_seconds
    assert record.workspace_bytes == baseline.workspace_bytes
    assert record.metadata["duration_kind"] == baseline.metadata["duration_kind"]
    assert record.metadata["memory_kind"] == baseline.metadata["memory_kind"]
    assert source.tasks["update_impl"].overlap_costs is None
    costed = apply_profiles(program(), merged)._compute_profiles()
    assert next(p for p in costed if p.profile_id == "update_impl").overlap_costs == (
        cost,
    )


def test_representative_routes_use_actual_sizes_and_planner_traffic_units():
    source = program()
    shards = {
        f"a{i}": {
            "padded_bytes": size,
            "shard_bytes": size // 2,
            "dtype": "torch.float32",
            "groups": ((0, 1),),
        }
        for i, size in enumerate((64, 128, 256, 512, 1024))
    }
    source = replace(source, bindings={**source.bindings, "weight_shards": shards})
    info = CommunicationInfo(
        1e9,
        1e9,
        topology={"node0": (0, 1)},
        measurements=tuple(
            {
                "operation": "all_gather",
                "ranks": (0, 1),
                "dtype": "torch.float32",
                "tensor_bytes": row["padded_bytes"],
                "inplace": True,
                "duration_seconds": row["padded_bytes"] / 1e9,
                "setup": {"gpu": "test"},
                "topology": {"node0": (0, 1)},
            }
            for row in shards.values()
        ),
    )
    source = replace(source, setup={"gpu": "test"})
    routes = overlap_routes(source, info)
    # Five sizes are within the limit, so each is probed.
    assert [r.buffer_bytes for r in routes] == [64, 128, 256, 512, 1024]
    assert [r.bytes for r in routes] == [32, 64, 128, 256, 512]
    assert len({r.timing_class for r in routes}) == 1


def test_a_wide_route_is_probed_across_its_whole_size_range():
    # Many small objects and a few large ones, as a model's sharded weights are:
    # the middle object by count is small, far below the largest.
    small = [28 + 512 * index for index in range(60)]
    large = [1 << 20, 8 << 20, 64 << 20, 512 << 20, 4032 << 20]
    sizes = sorted(small + large)
    assert sizes[len(sizes) // 2] < 1 << 20
    chosen = spread_sizes(sizes)
    assert len(chosen) <= MAX_ROUTE_SIZES and set(chosen) <= set(sizes)
    assert chosen[0] == sizes[0] and chosen[-1] == sizes[-1]
    assert chosen == sorted(set(chosen))
    # No two neighbouring probes are more than three even steps apart, so
    # nothing between 1 MiB and 4 GiB is left to one interpolation.
    step = (math.log1p(sizes[-1]) - math.log1p(sizes[0])) / (MAX_ROUTE_SIZES - 1)
    gaps = [math.log1p(b) - math.log1p(a) for a, b in pairwise(chosen)]
    assert max(gaps) <= 3 * step
    assert sum(1 << 20 <= size < 4032 << 20 for size in chosen) >= 3
    # Few sizes are all kept, and the choice does not depend on input order.
    assert spread_sizes([64, 128, 256]) == [64, 128, 256]
    # With three, the middle probe is the size nearest the middle of the range
    # in log bytes, not the middle object by count.
    middle = (math.log1p(sizes[0]) + math.log1p(sizes[-1])) / 2
    nearest = min(sizes, key=lambda size: abs(math.log1p(size) - middle))
    assert spread_sizes(sizes, 3) == [sizes[0], nearest, sizes[-1]]
    assert nearest > sizes[len(sizes) // 2]
