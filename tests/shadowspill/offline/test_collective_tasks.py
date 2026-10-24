"""Collective completion is a task dependency, with costs supplied at planning."""

from dataclasses import replace

import pytest

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    Parallelism,
    Plan,
    ProfiledProgram,
    Profiles,
    Program,
    TaskProfile,
    apply_profiles,
    plan_program,
    write_report,
)
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.communication.requests import collective_requests
from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    MutationSpec,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)
from shadowspill.reporting.rows import task_rows

from ._example import options


def collective_program(workspace=None):
    compute = ResourceSpec("rank0", ResourceKind.COMPUTE)
    communication = ResourceSpec("rank0", ResourceKind.COMMUNICATION)
    return Program(
        devices=(DeviceSpec("rank0", "worker", "cuda", 0),),
        alias_groups=tuple(
            AliasGroupSpec(name, "rank0", size)
            for name, size in (
                ("weight", 64),
                ("partial", 32),
                ("reduced", 16),
            )
        ),
        objects=tuple(
            ObjectSpec(name, name, 0, size)
            for name, size in (
                ("weight", 64),
                ("partial", 32),
                ("reduced", 16),
            )
        ),
        tasks=(
            TaskSpec(
                "begin",
                ResourceSpec("rank0", ResourceKind.CONTROL),
                "begin",
                requires_entrypoint=False,
            ),
            TaskSpec(
                "projection",
                compute,
                "projection",
                dependencies=("begin",),
                inputs=("weight",),
                outputs=("partial",),
            ),
            TaskSpec(
                "reduce",
                communication,
                "reduce",
                dependencies=("projection",),
                inputs=("partial",),
                outputs=("reduced",),
            ),
            TaskSpec(
                "tail",
                compute,
                "tail",
                dependencies=("projection",),
                inputs=("weight",),
            ),
            TaskSpec(
                "consume",
                compute,
                "consume",
                dependencies=("reduce", "tail"),
                inputs=("reduced", "weight"),
            ),
        ),
        task_definitions=(
            *(
                TaskDefinition(name, name, 0)
                for name in ("begin", "projection", "tail", "consume")
            ),
            TaskDefinition("reduce", "reduce", workspace),
        ),
        initial_residency=(ResidencySpec("weight", MemoryLocation.SPILL),),
        parallelism=Parallelism.dp(ranks_per_replica=2),
        setup={"hardware": "fixture"},
        fixed_device_bytes=0,
        bindings={
            "kind": "forward",
            "weight_shards": {},
            "collective_tasks": {
                "reduce": {
                    "operation": "reduce_scatter",
                    "groups": ((1, 0),),
                    "tensor_bytes": 32,
                    "dtype": "torch.float32",
                },
            },
        },
    )


def compute_profiles():
    return Profiles(
        setup={"hardware": "fixture"},
        tasks={
            "begin": TaskProfile(0),
            "projection": TaskProfile(2e-6),
            "tail": TaskProfile(20e-6),
            "consume": TaskProfile(3e-6),
        },
    )


def communication(duration):
    topology = {"node0": (0, 1)}
    return CommunicationInfo(
        1e9,
        1e9,
        fixed_device_bytes=0,
        topology=topology,
        measurements=(
            {
                "operation": "reduce_scatter",
                "ranks": (1, 0),
                "tensor_bytes": 32,
                "dtype": "torch.float32",
                "duration_seconds": duration,
                "setup": {"hardware": "fixture"},
                "topology": topology,
            },
        ),
    )


def test_collective_memory_can_be_deferred_to_communication_info():
    source = collective_program()
    profiles = compute_profiles()
    deferred = apply_profiles(source, profiles)
    assert deferred.task_profiles["reduce"].workspace_bytes is None
    with pytest.raises(ValueError, match="incomplete communication memory"):
        deferred._costed_ir(communication(1e-6))
    profiles.tasks["reduce"] = TaskProfile(workspace_bytes=8)
    result = apply_profiles(source, profiles)
    assert result.task_profiles["reduce"].duration_seconds is None
    assert "reduce" in result.estimated_memory
    with pytest.raises(ValueError, match="require CommunicationInfo"):
        result._costed_ir()
    profiles.tasks["reduce"] = TaskProfile(1e-6, 8)
    with pytest.raises(ValueError, match="through CommunicationInfo"):
        apply_profiles(source, profiles)


def test_planning_prices_completion_and_retains_buffers_until_consumed(tmp_path):
    source = collective_program(workspace=8)
    profiled = apply_profiles(source, compute_profiles())
    profiled.save(tmp_path / "profiled.ssp")
    profiled = ProfiledProgram.load(tmp_path / "profiled.ssp")
    assert profiled.task_profiles["reduce"].duration_seconds is None
    assert collective_requests(source, None)[0].ranks == (1, 0)
    results = [
        plan_program(
            profiled,
            memory_budgets=MemoryBudgets(256, 1024),
            communication_info=communication(duration),
            search_options=options(),
        )
        for duration in (10e-6, 40e-6)
    ]
    for result, duration in zip(results, (10e-6, 40e-6), strict=True):
        profiles = {row.profile_id: row for row in result._costed_program.profiles}
        assert profiles["reduce"].runtime_ns == round(duration * 1e9)
        assert profiles["reduce"].workspace_bytes == 8
        assert profiles["reduce"].resource_phases[0].resource == "sharded_collectives"
        intervals = {row.task_id: row for row in result.timeline.task_intervals}
        assert intervals["reduce"].start_ns >= intervals["projection"].end_ns
        assert intervals["consume"].start_ns >= intervals["reduce"].end_ns
        assert intervals["consume"].start_ns >= intervals["tail"].end_ns
        # Inputs, output and scratch of an in-flight reduction all consume GPU
        # capacity; changing its timing changes the simulation without profiling.
        assert intervals["reduce"].start_ns < intervals["tail"].end_ns
        assert intervals["tail"].start_ns < intervals["reduce"].end_ns
        assert result.summary["execution_peak_bytes"] == 64 + 32 + 16 + 8
        row = next(row for row in task_rows(result) if row["task_id"] == "reduce")
        assert row["resource"] == "communication"
        assert row["collective_cost_source"] == "measurement"
        assert row["collective_ring_equivalent_bytes"] == 16
        result.save(tmp_path / "plan.ssp")
        assert Plan.load(tmp_path / "plan.ssp").summary == result.summary
    assert (
        results[1].summary["duration_seconds"] > results[0].summary["duration_seconds"]
    )
    write_report(results[0], tmp_path / "report")
    assert (tmp_path / "report" / "timeline.svg").exists()


def test_inplace_weight_publication_waits_for_update_without_duplicate_full_storage():
    source = collective_program(workspace=8)
    source = replace(
        source,
        objects=(*source.objects, ObjectSpec("owned_weight", "weight", 0, 32)),
        tasks=(
            *source.tasks[:-1],
            replace(
                source.tasks[-1],
                inputs=("reduced", "owned_weight"),
                mutations=(MutationSpec("owned_weight"),),
            ),
            TaskSpec(
                "publish",
                ResourceSpec("rank0", ResourceKind.COMMUNICATION),
                "gather",
                dependencies=("consume",),
                inputs=("owned_weight", "weight"),
                mutations=(MutationSpec("weight"),),
            ),
        ),
        task_definitions=(
            *source.task_definitions,
            TaskDefinition("gather", "gather", 0),
        ),
        bindings={
            **source.bindings,
            "collective_tasks": {
                **source.bindings["collective_tasks"],
                "gather": {
                    "operation": "all_gather",
                    "groups": ((1, 0),),
                    "tensor_bytes": 64,
                    "dtype": "torch.float32",
                    "inplace": True,
                },
            },
        },
    )
    info = replace(
        communication(10e-6),
        scale_up_bytes_per_second=1e9,
        scale_up_latency_seconds=0,
    )
    plan = plan_program(
        apply_profiles(source, compute_profiles()),
        memory_budgets=MemoryBudgets(256, 1024),
        communication_info=info,
        search_options=options(),
    )
    times = {row.task_id: row for row in plan.timeline.task_intervals}
    assert times["publish"].start_ns >= times["consume"].end_ns
    assert plan.summary["execution_peak_bytes"] == 120
    with pytest.raises(ValueError, match="in-place gather views"):
        replace(
            source,
            objects=tuple(
                replace(item, size_bytes=16)
                if item.object_id == "owned_weight"
                else item
                for item in source.objects
            ),
        )


def test_collective_descriptions_must_match_local_task_and_group_geometry():
    source = collective_program(workspace=0)
    with pytest.raises(ValueError, match="communication resource"):
        replace(
            source,
            tasks=tuple(
                replace(task, resource=ResourceSpec("rank0", ResourceKind.COMPUTE))
                if task.task_id == "reduce"
                else task
                for task in source.tasks
            ),
        )
    description = source.bindings["collective_tasks"]["reduce"]
    with pytest.raises(ValueError, match="cover logical ranks"):
        replace(
            source,
            bindings={
                **source.bindings,
                "collective_tasks": {
                    "reduce": {**description, "groups": ((0, 2),)},
                },
            },
        )
    with pytest.raises(ValueError, match="bytes disagree"):
        replace(
            source,
            bindings={
                **source.bindings,
                "collective_tasks": {
                    "reduce": {**description, "tensor_bytes": 64},
                },
            },
        )
