"""Entry transfers are priced actions; declared device values are already present."""

from __future__ import annotations

from dataclasses import replace

import pytest

from shadowspill.errors import PlanInfeasibleError
from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryActionKind,
    MemoryLocation,
    MutationSpec,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import GenericPlanningOptions, pressurefit
from shadowspill.simulator import SimulationConfig


def _program() -> ShadowSpillProgram:
    return ShadowSpillProgram(
        devices=(DeviceSpec("device", "process", "test", 0),),
        alias_groups=(
            AliasGroupSpec("value", "device", 100, retain_spill_copy=True),
            AliasGroupSpec("empty", "device", 0),
        ),
        objects=(
            ObjectSpec("whole", "value", 0, 100),
            ObjectSpec("view", "value", 20, 30),
            ObjectSpec("empty", "empty", 0, 0),
        ),
        profiles=(
            TaskProfile("control", 0, 0, "control"),
            TaskProfile("work", 1000, 0, "abi"),
        ),
        tasks=(
            TaskSpec(
                "start",
                ResourceSpec("device", ResourceKind.CONTROL),
                "control",
                requires_entrypoint=False,
            ),
            TaskSpec(
                "update",
                ResourceSpec("device", ResourceKind.COMPUTE),
                "work",
                dependencies=("start",),
                inputs=("whole", "view", "empty"),
                mutations=(MutationSpec("whole"),),
            ),
        ),
    )


def _plan(program: ShadowSpillProgram, location: MemoryLocation):
    return pressurefit(
        program,
        initial_residency=(ResidencySpec("value", location),),
        final_residency=(ResidencySpec("value", MemoryLocation.SPILL),),
        config=SimulationConfig.single_device(
            "device",
            device_capacity_bytes=200,
            spill_capacity_bytes=200,
            fetch_solo_bandwidth_bytes_per_second=1_000_000_000,
            fetch_concurrent_bandwidth_bytes_per_second=(1_000_000_000),
            evict_solo_bandwidth_bytes_per_second=1_000_000_000,
            evict_concurrent_bandwidth_bytes_per_second=(1_000_000_000),
            fetch_latency_ns=25,
            evict_latency_ns=25,
        ),
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
        workers=1,
    )


@pytest.mark.parametrize("location", [MemoryLocation.SPILL, MemoryLocation.DEVICE])
def test_entry_and_terminal_copies_are_priced_once(location: MemoryLocation) -> None:
    result = _plan(_program(), location)
    result.schedule.validate(_program())
    assert result.schedule.initial_residency == (ResidencySpec("value", location),)
    fetches = [
        item for item in result.schedule.actions if item.kind is MemoryActionKind.FETCH
    ]
    if location is MemoryLocation.SPILL:
        assert (
            len(fetches) == 1
        )  # both views share this copy; zero-byte input needs none
        assert fetches[0].trigger_task_id == "start"
        assert result.simulation.makespan_ns == 1200  # fetch + update + writeback
    else:
        assert fetches == []
        assert result.simulation.makespan_ns == 1100  # update + writeback
    assert sum(item.bytes for item in result.simulation.transfer_intervals) == (
        200 if location is MemoryLocation.SPILL else 100
    )


def test_first_task_cannot_silently_promote_a_spill_input() -> None:
    program = _program()
    program = replace(program, tasks=(replace(program.tasks[1], dependencies=()),))
    with pytest.raises(PlanInfeasibleError, match="preceding fetch boundary"):
        _plan(program, MemoryLocation.SPILL)
