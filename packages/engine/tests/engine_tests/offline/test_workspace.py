from dataclasses import replace

from shadowspill.planning.workspace import transient_output_workspace
from shadowspill.task.allocations.events import TaskAllocationEvent as Event
from shadowspill.task.allocations.events import TaskAllocationOperation as Op

from ._example import profiled


def test_temporary_output_extents_are_deduplicated():
    manifest = {
        "storage_contract": {
            "output_views": [
                {"leaf_index": 0, "root_id": 0},
                {"leaf_index": 1, "root_id": 0},
                {"leaf_index": 2, "root_id": 1},
            ]
        },
        "root_allocations": [
            {"root_id": 0, "requested_bytes": 128},
            {"root_id": 1, "requested_bytes": 256},
        ],
    }
    assert transient_output_workspace((0, 1), manifest, {}) == 128
    assert transient_output_workspace((0, 1, 2), manifest, {}) == 384
    assert transient_output_workspace((), {}, {}) == 0


def test_observed_lifetimes_do_not_double_count_peak_workspace():
    trace = [
        Event(0, Op.ALLOCATE, 512, 512),
        Event(0, Op.FREE, 512, 512),
        Event(1, Op.ALLOCATE, 128, 256, (0,), (0,)),
        Event(2, Op.ALLOCATE, 256, 256),
    ]
    # The output and later scratch together fit the earlier scratch peak.
    assert (
        transient_output_workspace(
            (0,), {}, {"allocation_trace": [item.to_dict() for item in trace]}
        )
        == 0
    )
    trace[-1] = Event(2, Op.ALLOCATE, 512, 512)
    assert (
        transient_output_workspace(
            (0,), {}, {"allocation_trace": [item.to_dict() for item in trace]}
        )
        == 256
    )


def test_repeated_task_costs_keep_distinct_output_selections(monkeypatch):
    from shadowspill.planning import workspace

    source = profiled()
    begin, update, end = source.program.tasks
    second = replace(update, task_id="second", dependencies=("update",))
    third = replace(update, task_id="third", dependencies=("second",))
    source = replace(
        source,
        program=replace(
            source.program,
            tasks=(begin, update, second, third, replace(end, dependencies=("third",))),
            bindings={
                "transient_outputs": {"update": [0], "second": [1], "third": [0]}
            },
        ),
        task_profiles={
            **source.task_profiles,
            "update_impl": replace(
                source.task_profiles["update_impl"],
                metadata={
                    "allocation_trace": [
                        Event(0, Op.ALLOCATE, 128, 128, (0,), (0,)).to_dict(),
                        Event(1, Op.ALLOCATE, 256, 256, (1,), (1,)).to_dict(),
                    ]
                },
            ),
        },
    )
    calls = []

    def counted(leaves, manifest, metadata):
        if leaves:
            calls.append(tuple(leaves))
        return transient_output_workspace(leaves, manifest, metadata)

    monkeypatch.setattr(workspace, "transient_output_workspace", counted)
    costed = source._costed_ir()
    profiles = {profile.profile_id: profile for profile in costed.profiles}
    assert [profiles[task.profile_id].workspace_bytes for task in costed.tasks] == [
        0,
        144,
        272,
        144,
        0,
    ]
    assert calls == [(0,), (1,)]
    calls.clear()
    # A later derivation has its own evidence, not a process-global cache.
    changed = replace(
        source,
        task_profiles={
            **source.task_profiles,
            "update_impl": replace(
                source.task_profiles["update_impl"], workspace_bytes=32
            ),
        },
    )._costed_ir()
    assert {p.profile_id: p.workspace_bytes for p in changed.profiles}[
        "update_impl.workspace_128"
    ] == 160
    assert calls == [(0,), (1,)]
