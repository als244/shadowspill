"""Physical admission accounting and causal reuse simulator regressions."""

from __future__ import annotations

from reference.python.admission import replay_admission
from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectRole,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import AdmissionFacts, TaskAdmissionSpec
from shadowspill.pytorch.planning.admission import (
    simulation_admission_from_replay,
)
from shadowspill.simulator import (
    ActionPhysicalDelta,
    MemoryReuseDependency,
    SimulationAdmission,
    SimulationConfig,
    TaskPhysicalDelta,
    simulate,
)


def _program() -> Program:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    return Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec(
                "evicted_activation",
                "cuda_0",
                96,
                retain_spill_copy=True,
            ),
            AliasGroupSpec("fetched_activation", "cuda_0", 96),
        ),
        objects=(
            ObjectSpec(
                "old_activation",
                "evicted_activation",
                0,
                96,
                ObjectRole.ACTIVATION,
            ),
            ObjectSpec(
                "next_activation",
                "fetched_activation",
                0,
                96,
                ObjectRole.ACTIVATION,
            ),
        ),
        profiles=(TaskProfile("trigger_profile", 10, 0, "trigger_abi"),),
        tasks=(
            TaskSpec(
                "eviction_trigger",
                compute,
                "trigger_profile",
                inputs=("old_activation",),
            ),
            TaskSpec(
                "fetch_trigger",
                compute,
                "trigger_profile",
                dependencies=("eviction_trigger",),
            ),
            TaskSpec(
                "consumer",
                compute,
                "trigger_profile",
                dependencies=("fetch_trigger",),
                inputs=("next_activation",),
            ),
        ),
    )


def _schedule() -> MemorySchedule:
    return MemorySchedule(
        initial_residency=(
            ResidencySpec("evicted_activation", MemoryLocation.DEVICE),
            ResidencySpec("fetched_activation", MemoryLocation.SPILL),
        ),
        actions=(
            MemoryAction(
                "eviction_trigger",
                "evicted_activation",
                MemoryActionKind.OFFLOAD,
            ),
            MemoryAction(
                "fetch_trigger",
                "fetched_activation",
                MemoryActionKind.PREFETCH,
            ),
        ),
        final_residency=(
            ResidencySpec("evicted_activation", MemoryLocation.SPILL),
            ResidencySpec("fetched_activation", MemoryLocation.DEVICE),
        ),
    )


def _config() -> SimulationConfig:
    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=96,
        spill_capacity_bytes=192,
        fetch_bandwidth_bytes_per_second=1_000_000_000,
        evict_bandwidth_bytes_per_second=1_000_000_000,
    )


def _admission() -> SimulationAdmission:
    return SimulationAdmission(
        initial_physical_bytes=(("cuda_0", 96),),
        action_deltas=(
            ActionPhysicalDelta(0, 0, 0),
            ActionPhysicalDelta(1, 0, 0),
        ),
        task_deltas=(
            TaskPhysicalDelta("eviction_trigger", 0, 0),
            TaskPhysicalDelta("fetch_trigger", 0, 0),
            TaskPhysicalDelta("consumer", 0, 0),
        ),
        reuse_dependencies=(MemoryReuseDependency(0, successor_action_index=1),),
    )


def test_without_a_reuse_certificate_the_fetch_waits_for_the_eviction() -> None:
    """The certificate buys accounting, not feasibility.

    A fetch into space an eviction has not yet released simply waits for it,
    the way the runtime would. The two runs produce the same intervals and
    differ only in why the fetch waited: uncertified it is blocked on device
    capacity, certified it is blocked on the reuse itself. Only the certified
    run can hold both copies logically resident at once.
    """

    result = simulate(_program(), _schedule(), config=_config())

    eviction, fetch = result.transfer_intervals
    assert (eviction.start_ns, eviction.end_ns) == (10, 106)
    # Ready at its trigger, not at the moment room appeared: a wait that
    # showed up as a later ready time would hide the pressure that caused it.
    assert (fetch.ready_ns, fetch.start_ns, fetch.end_ns) == (20, 106, 202)
    assert fetch.stall_reasons == ("device-capacity",)
    assert result.makespan_ns == 212
    # Uncertified, the two copies are never logically resident together.
    assert result.device_peak("cuda_0").object_bytes == 96


def test_causal_reuse_preserves_peak_and_delays_wire_start() -> None:
    program = _program()
    replay = replay_admission(
        program,
        _schedule(),
        facts=AdmissionFacts(
            "cuda_0",
            96,
            96,
            1,
            tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
        ),
    )
    admission = simulation_admission_from_replay(
        replay,
        _program(),
        _schedule(),
    )
    assert admission == _admission()

    result = simulate(
        _program(),
        _schedule(),
        config=_config(),
        admission=admission,
    )
    eviction, fetch = result.transfer_intervals
    assert (eviction.start_ns, eviction.end_ns) == (10, 106)
    assert (fetch.ready_ns, fetch.start_ns, fetch.end_ns) == (20, 106, 202)
    assert fetch.stall_reasons == ("memory-reuse",)
    assert result.task_intervals[-1].start_ns == 202
    peak = result.device_peak("cuda_0")
    assert peak.object_bytes == 192
    assert peak.total_bytes == 96
    assert result.makespan_ns == 212
