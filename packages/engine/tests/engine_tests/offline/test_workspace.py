from shadowspill.planning.workspace import transient_output_workspace
from shadowspill.task.allocations.events import TaskAllocationEvent as Event
from shadowspill.task.allocations.events import TaskAllocationOperation as Op


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
