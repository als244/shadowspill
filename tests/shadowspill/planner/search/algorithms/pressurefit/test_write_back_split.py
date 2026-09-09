"""Clean early, release late: the evictions something waited on are split."""

from __future__ import annotations

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryActionKind,
    MemoryLocation,
    MutationSpec,
    ObjectRole,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    GenericPlanningOptions,
)
from shadowspill.planner.admission import AdmissionFacts, TaskAdmissionSpec
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import (
    PressureFitOptions,
)
from shadowspill.simulator import SimulationConfig

#: A copy of the state takes about 3.9 ms at the bandwidth below, which is
#: long enough to see in a makespan and long enough to make a fetch wait.
_LONG_TASK_NS = 40_000_000
_SHORT_TASK_NS = 1_000_000


def _program(*, retained: bool = True) -> ShadowSpillProgram:
    """One object written early, needed again late, evicted in between."""

    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    return ShadowSpillProgram(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("state_storage", "cuda_0", 4096, retain_spill_copy=retained),
            AliasGroupSpec("other_storage", "cuda_0", 4096, retain_spill_copy=True),
        ),
        objects=(
            ObjectSpec("state", "state_storage", 0, 4096, ObjectRole.OPTIMIZER_STATE),
            ObjectSpec("other", "other_storage", 0, 4096, ObjectRole.PARAMETER),
        ),
        profiles=(
            TaskProfile("write_profile", _SHORT_TASK_NS, 0, "write_abi"),
            TaskProfile("long_profile", _LONG_TASK_NS, 0, "long_abi"),
            TaskProfile("read_profile", _SHORT_TASK_NS, 0, "read_abi"),
        ),
        tasks=(
            TaskSpec(
                "write_state",
                compute,
                "write_profile",
                inputs=("state",),
                mutations=(MutationSpec("state"),),
            ),
            TaskSpec(
                "long",
                compute,
                "long_profile",
                dependencies=("write_state",),
                inputs=("state",),
            ),
            TaskSpec(
                "read_other",
                compute,
                "read_profile",
                dependencies=("long",),
                inputs=("other",),
            ),
            TaskSpec(
                "read_state",
                compute,
                "read_profile",
                dependencies=("read_other",),
                inputs=("state",),
            ),
        ),
    )


_INITIAL = (
    ResidencySpec("state_storage", MemoryLocation.DEVICE),
    ResidencySpec("other_storage", MemoryLocation.SPILL),
)
_GENERIC = GenericPlanningOptions(
    minimum_object_bytes_evict_eligible=0, deterministic=True
)


def _machine(device_capacity_bytes: int = 8000) -> SimulationConfig:
    """A slow enough lane to see the copy; by default, room for one object."""

    return SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=device_capacity_bytes,
        spill_capacity_bytes=1 << 20,
        fetch_bandwidth_bytes_per_second=1 << 20,
        evict_bandwidth_bytes_per_second=1 << 20,
    )


def _plan(program: ShadowSpillProgram, *, split: bool, **kwargs):
    # The option lives on the search, so varying it means a search built
    # with it rather than an argument passed beside one.
    search = PressureFit(PressureFitOptions(split_write_backs=split))
    return search(
        program,
        initial_residency=_INITIAL,
        config=kwargs.pop("config", None) or _machine(),
        generic=_GENERIC,
        **kwargs,
    )


def _kinds(plan) -> list[MemoryActionKind]:
    return [action.kind for action in plan.schedule.actions]


def test_an_eviction_something_waited_on_becomes_a_write_back_and_a_release() -> None:
    """The fetch of the other object cannot start until the state is gone, so
    the copy that evicts it is holding the program up."""

    whole = _plan(_program(), split=False)
    assert MemoryActionKind.EVICT in _kinds(whole)
    assert MemoryActionKind.WRITE_BACK not in _kinds(whole)

    split = _plan(_program(), split=True)
    kinds = _kinds(split)
    assert MemoryActionKind.WRITE_BACK in kinds
    assert MemoryActionKind.EVICT not in kinds

    # the copy moved off the boundary that evicted, so the plan is shorter by
    # about what the copy takes
    assert split.simulation.makespan_ns < whole.simulation.makespan_ns
    saved = whole.simulation.makespan_ns - split.simulation.makespan_ns
    assert saved > 3_000_000

    # the copy is triggered where the value was last written, and the release
    # it paid for stays where the eviction was
    actions = split.schedule.actions
    write_back = next(
        item for item in actions if item.kind is MemoryActionKind.WRITE_BACK
    )
    release = next(
        item
        for item in actions
        if item.kind is MemoryActionKind.RELEASE
        and item.alias_group_id == write_back.alias_group_id
    )
    assert write_back.alias_group_id == "state_storage"
    assert write_back.trigger_task_id == "write_state"
    order = [task.task_id for task in _program().tasks]
    assert order.index(write_back.trigger_task_id) < order.index(
        release.trigger_task_id
    )


def test_an_eviction_nothing_waited_on_is_left_whole() -> None:
    """With room for both objects the eviction only serves the residency the
    plan was asked to end in. Moving its copy would buy nothing."""

    final = (ResidencySpec("state_storage", MemoryLocation.SPILL),)
    roomy = _machine(device_capacity_bytes=16_000)
    whole = _plan(_program(), split=False, config=roomy, final_residency=final)
    split = _plan(_program(), split=True, config=roomy, final_residency=final)

    assert MemoryActionKind.EVICT in _kinds(whole)
    assert MemoryActionKind.WRITE_BACK not in _kinds(split)
    assert _kinds(split) == _kinds(whole)
    assert split.simulation.makespan_ns == whole.simulation.makespan_ns


def test_an_object_that_keeps_no_spill_copy_is_left_whole() -> None:
    """Releasing such an object frees its spill copy, so a split would throw
    away exactly what the write-back wrote."""

    program = _program(retained=False)
    whole = _plan(program, split=False)
    split = _plan(program, split=True)

    assert MemoryActionKind.EVICT in _kinds(whole)
    assert MemoryActionKind.WRITE_BACK not in _kinds(split)
    assert _kinds(split) == _kinds(whole)


def _pool(program: ShadowSpillProgram, pool_bytes: int = 1 << 20) -> AdmissionFacts:
    return AdmissionFacts(
        "cuda_0",
        pool_bytes,
        pool_bytes,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )


def test_a_split_plan_survives_physical_admission_and_placement() -> None:
    """A write-back has to pass the pool, not only the simulator.

    Admission counts it as a copy that retires no lease, and the lease it
    leaves alone is ended by the release. Without that, every plan carrying a
    write-back is refused before it can be placed, and the split is silently
    inert wherever a pool is configured -- which is everywhere real.
    """

    program = _program()
    facts = _pool(program)
    whole = _plan(program, split=False, placement=facts)
    split = _plan(program, split=True, placement=facts)

    assert MemoryActionKind.WRITE_BACK in _kinds(split)
    assert MemoryActionKind.EVICT not in _kinds(split)
    assert split.simulation.makespan_ns < whole.simulation.makespan_ns
