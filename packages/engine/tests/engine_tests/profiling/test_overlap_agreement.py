"""Paired probes agree on ordering without conflating rank-local kernel choices."""

from dataclasses import replace
from types import SimpleNamespace

from shadowspill.artifacts import TaskDefinition
from shadowspill.communication.overlap import OverlapRoute
from shadowspill.communication.requests import CollectiveRequest
from shadowspill.profiling.cache_identity import task_identities
from shadowspill.profiling.overlap import _probe_identity


def program(binary):
    component = {"profile_key": "form", "inputs": [], "outputs": []}
    return SimpleNamespace(
        task_definitions=(TaskDefinition("form", binary),),
        bindings={
            "training_templates": [[]],
            "optimizer_templates": [component],
            "task_inputs": {"form": {"count": 0}},
            "manifests": {"form": {}},
        },
        setup={"gpu": "fixture"},
        resources={},
    )


def test_finalized_binary_changes_do_not_change_probe_order_but_invalidate_cache():
    left, right = program("rank-0-kernel"), program("rank-1-kernel")
    jobs = [{"optimizer": left.bindings["optimizer_templates"][0]}]
    arguments = dict(stages=None, trials=1)
    assert _probe_identity(left, (), jobs, **arguments) == _probe_identity(
        right, (), jobs, **arguments
    )
    a = task_identities(left, jobs, options=None, environment={})
    b = task_identities(right, jobs, options=None, environment={})
    assert a["form"]["implementation"] == "rank-0-kernel"
    assert b["form"]["implementation"] == "rank-1-kernel"
    assert a != b


def test_probe_forms_order_collective_geometry_and_sampling_must_still_agree():
    value = program("binary")
    jobs = [{"optimizer": value.bindings["optimizer_templates"][0]}]
    jobs += [{"gradient": {"profile_key": "reduction", "inputs": [], "outputs": []}}]
    route = OverlapRoute(
        "collective",
        16,
        1e-6,
        (CollectiveRequest("all_reduce", (0, 1), 16, "torch.float32"),),
    )
    arguments = dict(stages=None, trials=1)
    baseline = _probe_identity(value, (route,), jobs, **arguments)
    assert baseline != _probe_identity(value, (route,), jobs[::-1], **arguments)
    assert baseline != _probe_identity(value, (route,), jobs, stages=None, trials=2)
    assert baseline != _probe_identity(value, (route,), jobs, stages=[0], trials=1)
    other_route = replace(
        route, requests=(replace(route.requests[0], ranks=(1, 0)),)
    )
    assert baseline != _probe_identity(value, (other_route,), jobs, **arguments)
    value.task_definitions = (TaskDefinition("different-form", "binary"),)
    assert baseline != _probe_identity(value, (route,), jobs, **arguments)
