"""The frontier a report draws: grouping, dominance and labels, without
rendering anything."""

from baselines.common.results import memory_peaks, row, summarize_cycles
from baselines.report import frontier, group_key, groups, label, point, slug, title


def completed(engine, tokens_per_second, device_gib, host_gib, **settings):
    measurement = summarize_cycles([[4096 / tokens_per_second]], global_tokens=4096)
    measurement["performance_result"] = settings.pop("performance_result", True)
    peaks = {
        "peak_allocated_bytes": int(device_gib * 2**30),
        "peak_reserved_bytes": int(device_gib * 2**30) + 1,
        "host_peak_rss_bytes": int(host_gib * 2**30),
    }
    return row(
        engine=engine,
        status="completed",
        model={"model": "llama3-8b", "implementation": "mlops", "mode": "full"},
        geometry={"sequence_length": 4096, "world_size": 1},
        precision={"model_dtype": "bfloat16"},
        configuration={"recompute": 1.0, "offload_policy": "none", **settings},
        hardware={"device": {"name": "RTX 5090", "count": 1}},
        provenance={},
        measurement=measurement,
        memory=memory_peaks([peaks]),
    )


def boundary(engine, **settings):
    return row(
        engine=engine,
        status="cuda_out_of_memory",
        model={"model": "llama3-8b", "implementation": "mlops", "mode": "full"},
        geometry={"sequence_length": 4096, "world_size": 1},
        precision={"model_dtype": "bfloat16"},
        configuration={"recompute": 1.0, "offload_policy": "none", **settings},
        hardware={"device": {"name": "RTX 5090", "count": 1}},
        provenance={},
        failure={"reason": "oom"},
    )


def test_the_frontier_keeps_what_nothing_beats_on_both_axes():
    points = [
        {"x": 10.0, "y": 100.0},
        {"x": 12.0, "y": 90.0},
        {"x": 14.0, "y": 150.0},
        {"x": 14.0, "y": 120.0},
        {"x": 20.0, "y": 150.0},
        {"x": 30.0, "y": 160.0},
    ]
    assert [item["x"] for item in frontier(points, "x", "y")] == [10.0, 14.0, 30.0]
    assert frontier([], "x", "y") == []


def test_only_completed_throughput_rows_are_points():
    assert point(completed("fsdp2", 1000.0, 8.0, 2.0))["device_gib"] == 8.0
    assert point(completed("fsdp2", 1000.0, 8.0, 2.0), "reserved")["device_gib"] > 8.0
    assert point(completed("fsdp2", 1000.0, 8.0, 2.0, performance_result=False)) is None
    assert point(boundary("fsdp2")) is None
    assert group_key(boundary("fsdp2")) == ("llama3-8b", "full", 4096, 0, "RTX 5090", 1)


def test_groups_hold_their_points_frontiers_boundary_and_varied_keys():
    rows = [
        completed("fsdp2", 1000.0, 8.0, 2.0, recompute=0.0),
        completed("fsdp2", 1200.0, 12.0, 2.0, recompute=0.5),
        completed("fsdp2", 900.0, 16.0, 2.0, recompute=1.0),
        completed("shadowspill", 1100.0, 6.0, 20.0, recompute=0.0),
        boundary("fsdp2", recompute=0.0, offload_policy="cpu"),
    ]
    result = groups(rows, "allocated")
    key = ("llama3-8b", "full", 4096, 0, "RTX 5090", 1)
    assert list(result) == [key]
    group = result[key]
    assert group["engines"] == ["fsdp2", "shadowspill"]
    assert group["keys"] == {
        "fsdp2": ["offload_policy", "recompute"],
        "shadowspill": [],
    }
    device = [item["device_gib"] for item in group["frontiers"]["fsdp2"]]
    assert device == [8.0, 12.0]
    assert len(group["boundary"]) == 1
    keys = group["keys"]["fsdp2"]
    assert label(group["boundary"][0], keys) == "offload_policy=cpu, recompute=0"
    assert slug(key) == "llama3-8b-full-4096-0-rtx-5090-1"
    assert title(key) == (
        "llama3-8b, full training, 4,096-token sequences, world size 1, RTX 5090"
    )
    lora = ("qwen35-27b", "lora", 4096, 16, "NVIDIA GeForce RTX 5090", 1)
    assert title(lora) == (
        "qwen35-27b, LoRA, 4,096-token sequences, 16 sequences per step, "
        "world size 1, NVIDIA GeForce RTX 5090"
    )


def test_a_boundary_row_without_a_device_joins_its_models_group():
    refused = boundary("shadowspill", execution_gib=4)
    refused["hardware"] = {}
    rows = [completed("shadowspill", 1000.0, 8.0, 2.0, execution_gib=8), refused]
    result = groups(rows, "reserved")
    (key,) = result
    assert key[4] == "RTX 5090"
    assert [item["settings"]["execution_gib"] for item in result[key]["boundary"]] == [
        4
    ]
    assert result[key]["keys"]["shadowspill"] == ["execution_gib"]
