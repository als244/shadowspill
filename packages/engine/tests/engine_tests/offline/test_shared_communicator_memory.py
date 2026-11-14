"""Measured startup belongs to shared groups and their warmed request union."""

from copy import deepcopy
from dataclasses import replace

import pytest

from shadowspill import CommunicationInfo
from shadowspill.planning.worker_memory import (
    communicator_reservation,
    shared_fixed_overhead,
)

from .test_collective_tasks import collective_program
from .test_communication_memory import measured_info
from .test_worker_memory import plan


def observation(*extra_operations, size=96):
    value = measured_info().to_dict()
    value["fixed_device_bytes"] = size
    for operation in extra_operations:
        row = deepcopy(value["measurements"][0])
        row["operation"] = operation
        value["measurements"].append(row)
    return CommunicationInfo.from_dict(value)


def worker_plan(info):
    result = plan(8, 2, 10, communication=info, scratch=3, distributed=True)
    result.program = collective_program()
    return result


def test_independent_timings_for_same_warmup_share_one_reservation(tmp_path):
    first = observation()
    second = replace(first, fetch_bytes_per_second=2e9, fixed_device_bytes=160)
    second.save(tmp_path / "independent.ssp")
    second = CommunicationInfo.load(tmp_path / "independent.ssp")
    plans = [worker_plan(first), worker_plan(second)]
    assert communicator_reservation(plans) == 160
    assert shared_fixed_overhead(plans) == 183
    assert communicator_reservation(list(reversed(plans)) * 2) == 160


def test_one_calibration_covers_request_union_and_keeps_largest_observation():
    superset = observation("all_gather", size=160)
    subset = observation(size=192)
    plans = [worker_plan(superset), worker_plan(subset)]
    assert communicator_reservation(plans) == 192
    assert shared_fixed_overhead(plans) == shared_fixed_overhead([plans[1]])


def test_incomparable_request_sets_still_add_startup_reservations():
    plans = [
        worker_plan(observation("all_gather", size=160)),
        worker_plan(observation("all_reduce", size=192)),
    ]
    assert communicator_reservation(plans) == 352


@pytest.mark.parametrize("difference", ["backend", "setup", "coverage", "memory"])
def test_no_shared_discount_without_matching_scope_and_coverage(difference):
    first = observation()
    value = observation(size=160).to_dict()
    if difference == "backend":
        backend = {"nccl_version": [2, 30, 0]}
        value["metadata"]["backend_setup"] = backend
        value["metadata"]["memory_scope"]["backend_setup"] = backend
        value["measurements"][0]["backend_setup"] = backend
    elif difference == "setup":
        value["measurements"][0]["setup"] = {"hardware": "different"}
    elif difference == "coverage":
        value["measurements"][0]["operation"] = "all_gather"
    else:
        del value["measurements"][0]["memory"]
    plans = [worker_plan(first), worker_plan(CommunicationInfo.from_dict(value))]
    assert communicator_reservation(plans) == 256


def test_one_shared_record_must_cover_every_plan_using_it():
    info = observation()
    first, second = worker_plan(info), worker_plan(info)
    bindings = dict(second.program.bindings)
    bindings["weight_shards"] = {
        "weight": {"groups": ((1, 0),), "padded_bytes": 64, "dtype": "torch.float32"}
    }
    second.program = replace(second.program, bindings=bindings)
    other = worker_plan(observation("all_reduce", size=160))
    assert communicator_reservation([first, second, other]) == 256
