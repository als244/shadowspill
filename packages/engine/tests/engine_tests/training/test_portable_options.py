"""Portable phase records preserve the existing engine's option meanings."""

from shadowspill.artifacts._values import encoded
from shadowspill.parallelism import Parallelism
from shadowspill.planner import SearchOptions
from shadowspill.task.profiling import ProfilingOptions
from shadowspill_client import PlanningOptions
from shadowspill_client import ProfilingOptions as PortableProfilingOptions
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.requests import ComputeRequest, RequestIdentity


def test_portable_profiling_defaults_decode_to_exact_existing_policy():
    portable = PortableProfilingOptions()
    assert ProfilingOptions.from_dict(portable.to_dict()) == ProfilingOptions()
    changed = PortableProfilingOptions(
        minimum_samples=5, maximum_samples=10, measurement_seconds=0.5
    )
    assert ProfilingOptions.from_dict(changed.to_dict()).measurement_seconds == 0.5


def test_planning_records_carry_existing_search_serialization():
    original = SearchOptions()
    portable = PlanningOptions(
        microbatch_counts=[1, 2], search=original.to_dict(), workers=2
    )
    restored = SearchOptions.from_dict(portable.to_dict()["search"], "search")
    assert restored.to_dict() == original.to_dict()
    parallelism = Parallelism.dp(ranks_per_replica=2)
    assert Parallelism.from_dict(parallelism.to_dict()) == parallelism


def test_wire_metadata_uses_engine_artifact_canonical_json_convention():
    request = ComputeRequest(
        RequestIdentity("request", "run", 1, "writer", 1, "server"), "optim_step"
    )
    assert encode(request) == encoded(
        {"version": 1, "kind": "compute.request", "value": request.to_dict()}
    )
