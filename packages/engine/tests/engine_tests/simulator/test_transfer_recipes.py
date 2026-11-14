"""Backing capacity, phase dependencies, and host/collective resource accounting."""

from __future__ import annotations

import ctypes
from dataclasses import replace

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
    TransferPhase,
    TransferSpec,
)
from shadowspill.simulator import SimulationConfig, simulate
from shadowspill.simulator.capi import CProgram, simulator_api
from shadowspill.simulator.indexing.template import index_simulation_template
from shadowspill.simulator.model import (
    DeviceSimulationConfig,
    SimulationInfeasibleError,
)

G = 1_000_000_000


def recipe(size=100, gather=60, scatter=80, resource="group"):
    return TransferSpec(
        size,
        (
            TransferPhase("copy", size),
            TransferPhase("timed", duration_ns=gather, resource=resource),
        ),
        (
            TransferPhase("timed", duration_ns=scatter, resource=resource),
            TransferPhase("copy", size),
        ),
    )


def case(*, fetching=True, evicting=False, fetch_recipe=None, evict_recipe=None):
    aliases = []
    initial = []
    actions = []
    final = []
    for name, enabled, direction, spec in (
        ("weight", fetching, MemoryActionKind.FETCH, fetch_recipe or recipe()),
        ("gradient", evicting, MemoryActionKind.EVICT, evict_recipe or recipe()),
    ):
        if not enabled:
            continue
        aliases.append(AliasGroupSpec(name, "gpu", 400, transfer=spec))
        fetch = direction == MemoryActionKind.FETCH
        initial.append(
            ResidencySpec(
                name, MemoryLocation.SPILL if fetch else MemoryLocation.DEVICE
            )
        )
        final.append(
            ResidencySpec(
                name, MemoryLocation.DEVICE if fetch else MemoryLocation.SPILL
            )
        )
        actions.append(MemoryAction("entry", name, direction))
    program = ShadowSpillProgram(
        (DeviceSpec("gpu", "rank", "cuda", 0),),
        tuple(aliases),
        tuple(ObjectSpec(a.alias_group_id, a.alias_group_id, 0, 400) for a in aliases),
        (TaskProfile("empty", 0, 0, "v1"), TaskProfile("use", 10, 0, "v1")),
        (
            TaskSpec("entry", ResourceSpec("gpu", ResourceKind.CONTROL), "empty"),
            TaskSpec(
                "consumer",
                ResourceSpec("gpu", ResourceKind.COMPUTE),
                "use",
                dependencies=("entry",),
                inputs=("weight",) if fetching else (),
            ),
        ),
    )
    schedule = MemorySchedule(tuple(initial), tuple(actions), tuple(final))
    config = SimulationConfig(
        (DeviceSimulationConfig("gpu", 800, G, G // 2, G, G // 2),), 200
    )
    return program, schedule, config


def test_fetch_waits_for_gather_and_charges_full_gpu_but_owned_cpu_storage():
    program, schedule, config = case()
    result = simulate(program, schedule, config=config)
    assert result.makespan_ns == 170
    assert result.spill_peak_bytes == 100
    assert result.device_peaks[0].object_bytes == 400
    assert result.transfer_intervals[0].bytes == 100
    assert result.task_intervals[-1].start_ns == 160
    with pytest.raises(SimulationInfeasibleError):
        simulate(program, schedule, config=replace(config, spill_capacity_bytes=99))
    with pytest.raises(SimulationInfeasibleError):
        simulate(
            program,
            schedule,
            config=replace(
                config, devices=(replace(config.devices[0], capacity_bytes=399),)
            ),
        )


def test_reduce_scatter_precedes_shard_offload():
    program, schedule, config = case(fetching=False, evicting=True)
    result = simulate(program, schedule, config=config)
    assert result.makespan_ns == 180
    assert result.spill_peak_bytes == 100
    assert result.transfer_intervals[0].bytes == 100


def test_collectives_do_not_occupy_host_copy_bandwidth():
    program, schedule, config = case(
        evicting=True, fetch_recipe=recipe(gather=100), evict_recipe=recipe(scatter=100)
    )
    result = simulate(program, schedule, config=config)
    assert {t.alias_group_id: t.end_ns for t in result.transfer_intervals} == {
        "weight": 200,
        "gradient": 200,
    }
    assert result.makespan_ns == 210


def test_collective_resource_waits_are_included_in_readiness():
    program, schedule, config = case(
        evicting=True,
        fetch_recipe=recipe(size=20, gather=200),
        evict_recipe=recipe(scatter=1000),
    )
    result = simulate(program, schedule, config=config)
    transfers = {t.alias_group_id: t for t in result.transfer_intervals}
    assert transfers["weight"].end_ns == 1200
    assert transfers["gradient"].end_ns == 1100
    assert "transfer-resource" in transfers["weight"].stall_reasons
    assert result.task_intervals[-1].start_ns == 1200


def test_ordered_collective_does_not_overtake_a_reduction_behind_a_copy():
    def ordered(spec):
        return replace(
            spec,
            fetch=tuple(
                replace(phase, submission_ordered=phase.kind == "timed")
                for phase in spec.fetch
            ),
            evict=tuple(
                replace(phase, submission_ordered=phase.kind == "timed")
                for phase in spec.evict
            ),
        )

    program, schedule, config = case(
        evicting=True,
        fetch_recipe=ordered(recipe(size=20, gather=200)),
        evict_recipe=ordered(recipe(scatter=1000)),
    )
    program = replace(
        program,
        alias_groups=(*program.alias_groups, AliasGroupSpec("ordinary", "gpu", 1000)),
        objects=(*program.objects, ObjectSpec("ordinary", "ordinary", 0, 1000)),
    )
    schedule = replace(
        schedule,
        initial_residency=(
            *schedule.initial_residency,
            ResidencySpec("ordinary", MemoryLocation.DEVICE),
        ),
        actions=(
            MemoryAction("entry", "ordinary", MemoryActionKind.EVICT),
            *schedule.actions,
        ),
        final_residency=(
            *schedule.final_residency,
            ResidencySpec("ordinary", MemoryLocation.SPILL),
        ),
    )
    config = replace(
        config,
        spill_capacity_bytes=2000,
        devices=(replace(config.devices[0], capacity_bytes=2000),),
    )
    result = simulate(program, schedule, config=config)
    transfers = {item.alias_group_id: item for item in result.transfer_intervals}
    # The initial 20-byte fetch shares bandwidth for 40 ns. The ordinary
    # eviction ends at 1020, then reduction [1020,2020] precedes the gather.
    assert transfers["ordinary"].end_ns == 1020
    assert transfers["gradient"].end_ns == 2120
    assert transfers["weight"].end_ns == 2220
    assert result.makespan_ns == 2230


def test_transfer_recipes_round_trip_and_change_program_identity():
    program, _, _ = case()
    assert ShadowSpillProgram.from_json(program.to_json()) == program
    ordinary = replace(
        program,
        alias_groups=tuple(replace(a, transfer=None) for a in program.alias_groups),
    )
    assert ordinary.digest != program.digest
    assert "transfer" not in ordinary.alias_groups[0].to_dict()


def test_retained_writeback_refuses_an_ordered_eviction_recipe():
    spec = recipe()
    spec = replace(
        spec,
        evict=(replace(spec.evict[0], submission_ordered=True), spec.evict[1]),
    )
    program, schedule, config = case(fetching=False, evicting=True, evict_recipe=spec)
    schedule = replace(
        schedule,
        actions=(MemoryAction("entry", "gradient", MemoryActionKind.WRITE_BACK),),
        final_residency=(ResidencySpec("gradient", MemoryLocation.DEVICE),),
    )
    with pytest.raises(SimulationInfeasibleError):
        simulate(program, schedule, config=config)


def test_pressurefit_uses_the_complete_isolated_recipe_cost():
    program, _, config = case()
    template = index_simulation_template(program, (), config)
    function = simulator_api().shadowspill_transfer_duration_ns
    function.argtypes = [
        ctypes.POINTER(CProgram),
        ctypes.c_uint32,
        ctypes.c_uint8,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    function.restype = ctypes.c_int
    duration = ctypes.c_uint64()
    assert (
        function(ctypes.byref(template.program), 0, 0, G, ctypes.byref(duration)) == 0
    )
    assert duration.value == 160
    assert (
        function(ctypes.byref(template.program), 0, 1, G, ctypes.byref(duration)) == 0
    )
    assert duration.value == 180
