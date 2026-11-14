"""Route identity, traffic coordinates and profile overlays stay CPU-only."""

from dataclasses import replace

from shadowspill import CommunicationInfo, Profiles, TaskProfile, apply_profiles
from shadowspill.communication.overlap import overlap_routes, timing_class
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
    assert [r.buffer_bytes for r in routes] == [64, 256, 1024]
    assert [r.bytes for r in routes] == [32, 128, 512]
    assert len({r.timing_class for r in routes}) == 1
