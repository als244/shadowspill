"""Do not evict the conditioned task's allocator workspace during timing."""

from contextlib import nullcontext
from types import SimpleNamespace

from shadowspill import Profiles, TaskProfile
from shadowspill.profiling import overlap


def test_admission_precedes_single_buffer_and_warmup_without_route_cache_flushes(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(overlap.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        overlap.torch.cuda, "empty_cache", lambda: calls.append("clear")
    )
    monkeypatch.setattr(
        overlap.torch.cuda, "mem_get_info", lambda device: ((32 << 20) + 288, 1 << 30)
    )

    class Buffer:
        def __init__(self, size, stream):
            calls.append(("buffer", size))

        def close(self):
            calls.append("close")

    monkeypatch.setattr(overlap, "ProbeWorkspace", Buffer)
    monkeypatch.setattr(
        overlap,
        "task_collective_count",
        lambda invoke, stream: calls.append("warm") or 0,
    )
    used_buffers = []

    def measure(invoke, stream, runtime, route, groups, **kwargs):
        assert calls.index("clear") < calls.index("warm")
        assert calls.count("clear") == 1
        used_buffers.append(kwargs["workspace"])
        calls.append(("probe", route.bytes))
        return {"rates": (0.8, 0.5), "logical_rank": 0}

    monkeypatch.setattr(overlap, "measure_route", measure)
    value = object.__new__(overlap._OverlapSession)
    value.profiles, value.cached_routes = Profiles(), {}
    value.baseline = {
        "task": TaskProfile(
            0.01, 128, "implementation", metadata={"warmup_persistent_bytes": [16]}
        )
    }
    value.routes = tuple(
        SimpleNamespace(timing_class="collective_test", bytes=n, buffer_bytes=n)
        for n in (64, 128, 144)
    )
    value.groups = SimpleNamespace(
        gather=lambda value: [value], check=lambda error: None
    )
    value.device, value.streams = "cuda", {"task": object()}
    value.function = lambda key: lambda *args: None
    value.trials, value.overlap_cache = None, None
    value.probe_baselines, value.probe_seconds = None, 0
    value.communication_reserve_bytes = 16
    value.measure("task", ())
    assert calls == [
        "clear",
        ("buffer", 128),
        "warm",
        ("probe", 64),
        ("probe", 128),
        "close",
    ]
    assert len(used_buffers) == 2 and used_buffers[0] is used_buffers[1]
    profile = value.profiles.tasks["task"]
    assert len(profile.overlap_costs) == 2
    rows = profile.metadata["overlap_calibration"]["observations"]
    assert sum(row["status"] == "insufficient_device_memory" for row in rows) == 1
