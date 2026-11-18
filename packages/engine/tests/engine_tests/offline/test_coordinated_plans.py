"""Different GPU schedules must agree before collective execution is admitted."""

from dataclasses import replace

import pytest

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    Plan,
    apply_profiles,
    plan_program,
)
from shadowspill.artifacts import Program, TaskDefinition
from shadowspill.artifacts.profiles import Profiles, TaskProfile
from shadowspill.errors import PlanSearchExhaustedError
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
    TaskSpec,
)
from shadowspill.parallelism import Parallelism
from shadowspill.planning.collectives import collective_sequence
from shadowspill.planning.coordination import coordinate_plans
from shadowspill.planning.costs import simulation_inputs
from shadowspill.planning.memory_search.admission.layout import (
    build_fixed_layout_admission,
)
from shadowspill.planning.memory_search.result import ProgramPlanResult
from shadowspill.simulator import simulate
from shadowspill.task.allocations.contract import TaskAllocationContract
from shadowspill.task.allocations.events import TaskAllocationEvent as Event
from shadowspill.task.allocations.events import TaskAllocationOperation as Op
from shadowspill.task.layout import reconcile_compiled_task_layout
from shadowspill.task.profiles import TaskMeasurement
from shadowspill.task.storage import TaskStorageContract

from ._example import options


def worker_plan(gpu, *, size, workspace, duration, order, serial=False, capacity=4096):
    """Two read-only sharded weights, local workspace, explicit fetch order."""
    storage = TaskStorageContract.build((), (), ())
    trace = (
        Event(0, Op.ALLOCATE, workspace, workspace),
        Event(0, Op.FREE, workspace, workspace),
    )
    measurement = TaskMeasurement(
        duration,
        workspace,
        workspace,
        (),
        (duration,),
        "static",
        allocation_trace=trace,
    )
    memory = {
        "memory_kind": "static",
        "allocation_trace": [event.to_dict() for event in trace],
        "allocation_contract": TaskAllocationContract.capture(trace, storage).to_dict(),
        "compiled_layout": reconcile_compiled_task_layout(
            storage, measurement
        ).identity(),
        "output_input_bindings": [],
        "off_device_output_leaves": [],
    }
    compute = ResourceSpec("gpu", ResourceKind.COMPUTE)
    entries = tuple(ResidencySpec(name, MemoryLocation.SPILL) for name in ("a", "b"))
    program = Program(
        devices=(DeviceSpec("gpu", "worker", "cuda", 0),),
        alias_groups=tuple(
            AliasGroupSpec(name, "gpu", size, retain_spill_copy=True)
            for name in ("a", "b")
        ),
        objects=tuple(ObjectSpec(name, name, 0, size) for name in ("a", "b")),
        tasks=(
            TaskSpec(
                "begin",
                compute,
                "invocation_start",
                phase="control",
                requires_entrypoint=False,
            ),
            TaskSpec("compute_a", compute, gpu, inputs=("a",), dependencies=("begin",)),
            TaskSpec(
                "compute_b", compute, gpu, inputs=("b",), dependencies=("compute_a",)
            ),
        ),
        task_definitions=(
            TaskDefinition("invocation_start", "start", 0),
            TaskDefinition(gpu, gpu),
        ),
        initial_residency=entries,
        final_residency=entries,
        setup={"gpu": gpu},
        parallelism=Parallelism.dp(ranks_per_replica=2),
        executables={gpu: gpu.encode()},
        bindings={
            "kind": "forward",
            "registrations": [
                {"name": name, "object_id": name, "parameter": True}
                for name in ("a", "b")
            ],
            "entrypoints": [
                {"task_id": "compute_" + name, "input_slots": [], "output_slots": []}
                for name in ("a", "b")
            ],
            "manifests": {
                gpu: {"storage_contract": storage.to_dict(), "root_allocations": []}
            },
            "weight_shards": {
                name: {
                    "groups": ((0, 1),),
                    "shard_bytes": size // 2,
                    "logical_bytes": size,
                    "padded_bytes": size,
                    "dtype": "torch.uint8",
                }
                for name in ("a", "b")
            },
        },
    )
    profiles = Profiles(
        tasks={
            "invocation_start": TaskProfile(0, 0, metadata={"memory_kind": "static"}),
            gpu: TaskProfile(duration / 1e9, workspace, metadata=memory),
        },
        fixed_device_bytes=0,
        setup=program.setup,
        metadata={"memory_kind": "static"},
    )
    info = CommunicationInfo(
        1e9,
        1e9,
        fixed_device_bytes=0,
        topology={"workers": (0, 1)},
        scale_up_bytes_per_second=1e9,
        scale_up_latency_seconds=1e-6,
        metadata={"memory_kind": "static"},
        measurements=(
            {
                "operation": "all_gather",
                "ranks": (0, 1),
                "tensor_bytes": size,
                "dtype": "torch.uint8",
                "inplace": True,
                "setup": program.setup,
                "topology": {"workers": (0, 1)},
                "memory": {"workspace_bytes": 0, "memory_kind": "static"},
            },
        ),
    )
    base = plan_program(
        apply_profiles(program, profiles),
        memory_budgets=MemoryBudgets(capacity, 4096),
        communication_info=info,
        search_options=options(),
    )
    assert base.memory_layout is not None
    schedule = MemorySchedule(
        entries,
        tuple(MemoryAction("begin", name, MemoryActionKind.FETCH) for name in order)
        + tuple(
            MemoryAction("compute_" + name, name, MemoryActionKind.RELEASE)
            for name in ("a", "b")
        ),
        entries,
    )
    if serial:
        schedule = replace(
            schedule,
            actions=(
                MemoryAction("begin", "a", MemoryActionKind.FETCH),
                MemoryAction("compute_a", "a", MemoryActionKind.RELEASE),
                MemoryAction("compute_a", "b", MemoryActionKind.FETCH),
                MemoryAction("compute_b", "b", MemoryActionKind.RELEASE),
            ),
        )
    costed, config = simulation_inputs(base.profiled_program, base.memory_budgets, info)
    result = ProgramPlanResult(
        costed,
        options(),
        entries,
        entries,
        config,
        schedule,
        (),
        simulate(costed, schedule, config=config),
        base.diagnostics,
    )
    certified = build_fixed_layout_admission(result, base._admission_facts)
    return replace(
        base,
        schedule=schedule,
        timeline=certified.simulation,
        memory_layout=certified.layout,
    )


@pytest.fixture
def workers():
    return {
        0: worker_plan(
            "fast", size=256, workspace=256, duration=1000, order=("a", "b")
        ),
        1: worker_plan(
            "slow", size=256, workspace=512, duration=2000, order=("b", "a")
        ),
    }


def test_common_schedule_keeps_local_kernels_costs_and_physical_admission(
    workers, tmp_path
):
    assert collective_sequence(workers[0]) != collective_sequence(workers[1])
    result = coordinate_plans(workers)
    assert collective_sequence(result[0]) == collective_sequence(result[1])
    assert result[0].schedule == result[1].schedule
    assert result[0].timeline.makespan_ns != result[1].timeline.makespan_ns
    assert (
        result[0].memory_layout.required_bytes != result[1].memory_layout.required_bytes
    )
    for rank, plan in result.items():
        original = workers[rank]
        assert plan.program is original.program
        assert plan.profiled_program is original.profiled_program
        assert plan._admission_facts is original._admission_facts
        assert plan.memory_layout.program_digest == original._costed_program.digest
        assert plan.memory_layout.schedule_digest == plan.schedule.digest
        assert not plan.simulation_only_reasons
        path = tmp_path / f"worker-{rank}.ssp"
        plan.save(path)
        loaded = Plan.load(path)
        assert loaded.memory_layout == plan.memory_layout
        assert loaded.diagnostics == plan.diagnostics
    assert coordinate_plans(result) == result


def test_accumulation_paths_coordinate_even_when_fresh_collectives_already_match(
    workers,
):
    def backward(plan, accumulated):
        return replace(
            plan,
            profiled_program=replace(
                plan.profiled_program,
                program=replace(
                    plan.program,
                    bindings={
                        **plan.program.bindings,
                        "kind": "forward_backward",
                        "distributed_training_lowered": "gradients",
                        "accumulate_gradients": accumulated,
                    },
                ),
            ),
        )

    common = coordinate_plans(workers)
    accumulated = {rank: backward(plan, True) for rank, plan in workers.items()}
    paths = {
        rank: replace(backward(common[rank], False), _accumulation_plan=plan)
        for rank, plan in accumulated.items()
    }
    result = coordinate_plans(paths)
    assert collective_sequence(result[0]) == collective_sequence(result[1])
    assert collective_sequence(result[0]._accumulation_plan) == collective_sequence(
        result[1]._accumulation_plan
    )
    assert result[0]._accumulation_plan.program is accumulated[0].program


def test_common_schedule_must_fit_the_smaller_worker():
    large = worker_plan(
        "large", size=256, workspace=256, duration=1000, order=("b", "a")
    )
    small = worker_plan(
        "small",
        size=256,
        workspace=512,
        duration=2000,
        order=("a", "b"),
        serial=True,
        capacity=768,
    )
    assert collective_sequence(large) != collective_sequence(small)
    result = coordinate_plans({0: large, 1: small})
    # Prefetching both weights needs 1024 bytes on the small GPU; its serial
    # schedule needs exactly 768. Only the latter can coordinate both workers.
    assert result[0].schedule == small.schedule
    assert result[1] is small
    assert result[1].memory_layout.required_bytes == 768


def test_no_common_candidate_is_search_exhaustion_not_an_infeasibility_proof(workers):
    other = workers[1]
    # A different task graph cannot inherit boundaries merely because its
    # ordinal alias names look the same.
    tasks = tuple(
        replace(t, phase="different") if t.task_id == "compute_a" else t
        for t in other._costed_program.tasks
    )
    changed = replace(
        other,
        _costed_program=replace(other._costed_program, tasks=tasks),
        memory_layout=None,
        _admission_facts=None,
    )
    with pytest.raises(
        PlanSearchExhaustedError, match="no searched schedule"
    ) as caught:
        coordinate_plans({0: workers[0], 1: changed})
    assert all("topology" in str(row) for row in caught.value.diagnostics)


def test_worker_names_are_not_rebound_to_a_different_group(workers):
    from shadowspill.training.preparation.planning.coordination import (
        coordinate_workers,
    )
    from shadowspill.training.preparation.planning.records import WorkerPlans

    originals = {rank: {"local": plan} for rank, plan in workers.items()}
    uses = dict.fromkeys(workers, WorkerPlans("local", None))
    names = WorkerPlans("coordinated", None)
    selected = coordinate_workers(uses, originals, names)
    assert selected == dict.fromkeys(workers, names)
    for rank in workers:
        assert originals[rank]["local"] is workers[rank]
    first = {rank: originals[rank]["coordinated"] for rank in workers}
    coordinate_workers(uses, originals, names)
    assert all(originals[rank]["coordinated"] is first[rank] for rank in workers)
