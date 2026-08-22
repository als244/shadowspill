from __future__ import annotations

import pytest

from reference.python.admission import replay_admission
from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    AdmissionTopology,
    PressureFitOptions,
    PressureFitSearchExhaustedError,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
    pressurefit,
)
from shadowspill.planner._admission import (
    compile_admission_topology,
    encode_schedule,
    evaluate_schedule_admission,
)
from shadowspill.planner._capi import planner_library_path
from shadowspill.pytorch.planning.admission import (
    simulation_admission_from_replay,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator._compiled import compile_simulation_template
from tests.shadowspill.planner._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)
from tests.shadowspill.simulator.test_admission_accounting import (
    _config as causal_config,
)
from tests.shadowspill.simulator.test_admission_accounting import (
    _program as causal_program,
)
from tests.shadowspill.simulator.test_admission_accounting import (
    _schedule as causal_schedule,
)

pytestmark = pytest.mark.skipif(
    planner_library_path() is None,
    reason="compiled planner library is not installed",
)


def _task_admission(
    task_id: str,
    *,
    workspace_extents: tuple[int, ...] = (),
    output_extents: tuple[tuple[str, int], ...] = (),
) -> TaskAdmissionSpec:
    """Build explicit physical evidence for one hand-authored test task."""

    steps: list[TaskAllocationStep] = []
    workspace_ordinals: list[int] = []
    for extent in workspace_extents:
        ordinal = len(steps)
        workspace_ordinals.append(ordinal)
        steps.append(
            TaskAllocationStep(
                ordinal,
                TaskAllocationStepKind.ALLOCATE,
                extent,
            )
        )
    for alias_id, extent in output_extents:
        steps.append(
            TaskAllocationStep(
                len(steps),
                TaskAllocationStepKind.ALLOCATE,
                extent,
                alias_id,
            )
        )
    steps.extend(
        TaskAllocationStep(ordinal, TaskAllocationStepKind.RELEASE)
        for ordinal in workspace_ordinals
    )
    return TaskAdmissionSpec(
        task_id,
        workspace_extents=workspace_extents,
        fresh_output_aliases=tuple(alias_id for alias_id, _ in output_extents),
        allocation_steps=tuple(steps),
    )


def _causal_topology() -> AdmissionTopology:
    program = causal_program()
    return AdmissionTopology(
        "cuda_0",
        96,
        96,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )


def test_compiled_selected_admission_matches_python_oracle() -> None:
    program = causal_program()
    schedule = causal_schedule()
    topology = _causal_topology()
    template = compile_simulation_template(program, (), causal_config())

    compiled = evaluate_schedule_admission(
        template,
        compile_admission_topology(topology, template),
        encode_schedule(schedule, template),
    )
    replay = replay_admission(
        program,
        schedule,
        topology=topology,
    )
    reference = simulation_admission_from_replay(
        replay,
        program,
        schedule,
        device_capacity_bytes=96,
    )

    assert compiled.simulation_admission == reference
    assert compiled.decision_digest == replay.pool.decision_digest
    assert compiled.peak_allocated_bytes == replay.pool.peak_allocated_bytes
    assert compiled.peak_reserved_bytes == replay.pool.peak_reserved_bytes
    assert compiled.peak_fragmentation_bytes == replay.pool.peak_fragmentation_bytes


def test_compiled_after_task_release_to_fetch_matches_python_oracle() -> None:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("released", "cuda_0", 64),
            AliasGroupSpec("fetched", "cuda_0", 64),
        ),
        objects=(
            ObjectSpec("released_object", "released", 0, 64),
            ObjectSpec("fetched_object", "fetched", 0, 64),
        ),
        profiles=(TaskProfile("profile", 1, 0, "abi"),),
        tasks=(
            TaskSpec(
                "boundary",
                compute,
                "profile",
                inputs=("released_object",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=(
            ResidencySpec("released", MemoryLocation.DEVICE),
            ResidencySpec("fetched", MemoryLocation.HOST),
        ),
        actions=(
            MemoryAction("boundary", "released", MemoryActionKind.RELEASE),
            MemoryAction("boundary", "fetched", MemoryActionKind.PREFETCH),
        ),
        final_residency=(ResidencySpec("fetched", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=64,
        host_capacity_bytes=128,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    template = compile_simulation_template(program, (), config)
    topology = AdmissionTopology(
        "cuda_0",
        64,
        64,
        1,
        (TaskAdmissionSpec("boundary"),),
    )

    compiled = evaluate_schedule_admission(
        template,
        compile_admission_topology(topology, template),
        encode_schedule(schedule, template),
    )
    replay = replay_admission(
        program,
        schedule,
        topology=topology,
    )
    reference = simulation_admission_from_replay(
        replay,
        program,
        schedule,
        device_capacity_bytes=64,
    )

    assert compiled.decision_digest == replay.pool.decision_digest
    assert compiled.simulation_admission == reference
    assert compiled.peak_allocated_bytes == replay.pool.peak_allocated_bytes
    assert compiled.peak_fragmentation_bytes == replay.pool.peak_fragmentation_bytes


def test_pressurefit_publishes_the_same_admission_aware_selected_result() -> None:
    program = training_chain_program(2)
    object_alias = {item.object_id: item.alias_group_id for item in program.objects}
    profiles = {item.profile_id: item for item in program.profiles}
    topology = AdmissionTopology(
        "cuda_0",
        512,
        224,
        1,
        tuple(
            _task_admission(
                task.task_id,
                workspace_extents=(
                    (profiles[task.profile_id].workspace_bytes,)
                    if profiles[task.profile_id].workspace_bytes
                    else ()
                ),
                output_extents=tuple(
                    (
                        alias_id,
                        next(
                            item.size_bytes
                            for item in program.alias_groups
                            if item.alias_group_id == alias_id
                        ),
                    )
                    for alias_id in dict.fromkeys(
                        object_alias[item] for item in task.outputs
                    )
                ),
            )
            for task in program.tasks
        ),
    )

    result = pressurefit(
        program,
        initial_residency=training_chain_initial(2),
        config=training_chain_config(224),
        admission=topology,
        options=PressureFitOptions(workers=1),
    )

    assert result.simulation.device_peak("cuda_0").total_bytes > 224
    assert result.simulation.device_peak("cuda_0").total_bytes <= 512
    assert result.diagnostics.selected_makespan_ns == result.simulation.makespan_ns


def test_compiled_admission_places_workspace_across_fragmented_ranges() -> None:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=tuple(
            AliasGroupSpec(alias, "cuda_0", 32) for alias in ("a", "b", "c")
        ),
        objects=tuple(ObjectSpec(alias, alias, 0, 32) for alias in ("a", "b", "c")),
        profiles=(TaskProfile("profile", 1, 64, "abi"),),
        tasks=(
            TaskSpec("release_middle", compute, "profile", inputs=("b",)),
            TaskSpec(
                "use_workspace",
                compute,
                "profile",
                dependencies=("release_middle",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=tuple(
            ResidencySpec(alias, MemoryLocation.DEVICE) for alias in ("a", "b", "c")
        ),
        actions=(MemoryAction("release_middle", "b", MemoryActionKind.RELEASE),),
        final_residency=(
            ResidencySpec("a", MemoryLocation.DEVICE),
            ResidencySpec("c", MemoryLocation.DEVICE),
        ),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=128,
        host_capacity_bytes=1,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    template = compile_simulation_template(program, (), config)

    def evaluate(extents: tuple[int, ...]):
        topology = AdmissionTopology(
            "cuda_0",
            128,
            128,
            1,
            (
                TaskAdmissionSpec("release_middle"),
                _task_admission("use_workspace", workspace_extents=extents),
            ),
        )
        return evaluate_schedule_admission(
            template,
            compile_admission_topology(topology, template),
            encode_schedule(schedule, template),
        )

    admitted = evaluate((32, 32))

    assert admitted.peak_allocated_bytes == 128
    with pytest.raises(ValueError, match="dynamic MemoryPool admission"):
        evaluate((64,))


def test_compiled_admission_sizes_reuse_results_independently_of_events() -> None:
    """One completion event may make several predecessor ranges reusable."""

    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    aliases = ("first", "second", "third")
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=tuple(AliasGroupSpec(alias, "cuda_0", 8) for alias in aliases),
        objects=tuple(ObjectSpec(alias, alias, 0, 8) for alias in aliases),
        profiles=(
            TaskProfile("release_profile", 1, 0, "release_abi"),
            TaskProfile("workspace_profile", 1, 24, "workspace_abi"),
        ),
        tasks=(
            TaskSpec(
                "release_all",
                compute,
                "release_profile",
                inputs=aliases,
            ),
            TaskSpec(
                "use_workspace",
                compute,
                "workspace_profile",
                dependencies=("release_all",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=tuple(
            ResidencySpec(alias, MemoryLocation.DEVICE) for alias in aliases
        ),
        actions=tuple(
            MemoryAction("release_all", alias, MemoryActionKind.RELEASE)
            for alias in aliases
        ),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=24,
        host_capacity_bytes=1,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    topology = AdmissionTopology(
        "cuda_0",
        24,
        24,
        1,
        (
            TaskAdmissionSpec("release_all"),
            _task_admission("use_workspace", workspace_extents=(24,)),
        ),
    )
    template = compile_simulation_template(program, (), config)

    compiled = evaluate_schedule_admission(
        template,
        compile_admission_topology(topology, template),
        encode_schedule(schedule, template),
    )
    replay = replay_admission(
        program,
        schedule,
        topology=topology,
    )

    assert len(replay.pool.dependencies) == 3
    assert compiled.decision_digest == replay.pool.decision_digest
    assert compiled.peak_allocated_bytes == 24


def test_compiled_admission_preserves_profiled_task_allocation_order() -> None:
    """A peak multiset alone can invent fragmentation that execution avoids."""

    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    sizes = {"hole_6": 6, "separator": 2, "hole_10": 10, "tail": 2, "out": 6}
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=tuple(
            AliasGroupSpec(alias, "cuda_0", size) for alias, size in sizes.items()
        ),
        objects=tuple(
            ObjectSpec(alias, alias, 0, size) for alias, size in sizes.items()
        ),
        profiles=(TaskProfile("profile", 1, 10, "abi"),),
        tasks=(
            TaskSpec(
                "release_holes",
                compute,
                "profile",
                inputs=("hole_6", "hole_10"),
            ),
            TaskSpec(
                "ordered_task",
                compute,
                "profile",
                dependencies=("release_holes",),
                outputs=("out",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=tuple(
            ResidencySpec(alias, MemoryLocation.DEVICE)
            for alias in ("hole_6", "separator", "hole_10", "tail")
        ),
        actions=(
            MemoryAction("release_holes", "hole_6", MemoryActionKind.RELEASE),
            MemoryAction("release_holes", "hole_10", MemoryActionKind.RELEASE),
        ),
        final_residency=tuple(
            ResidencySpec(alias, MemoryLocation.DEVICE)
            for alias in ("separator", "tail", "out")
        ),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=20,
        host_capacity_bytes=1,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    template = compile_simulation_template(program, (), config)
    ordered = TaskAdmissionSpec(
        "ordered_task",
        workspace_extents=(4, 6),
        fresh_output_aliases=("out",),
        allocation_steps=(
            TaskAllocationStep(
                0,
                TaskAllocationStepKind.ALLOCATE,
                6,
                "out",
            ),
            TaskAllocationStep(1, TaskAllocationStepKind.ALLOCATE, 4),
            TaskAllocationStep(2, TaskAllocationStepKind.ALLOCATE, 6),
            TaskAllocationStep(1, TaskAllocationStepKind.RELEASE),
            TaskAllocationStep(2, TaskAllocationStepKind.RELEASE),
        ),
    )

    def evaluate(task: TaskAdmissionSpec):
        topology = AdmissionTopology(
            "cuda_0",
            20,
            20,
            1,
            (TaskAdmissionSpec("release_holes"), task),
        )
        return evaluate_schedule_admission(
            template,
            compile_admission_topology(topology, template),
            encode_schedule(schedule, template),
        )

    admitted = evaluate(ordered)
    ordered_topology = AdmissionTopology(
        "cuda_0",
        20,
        20,
        1,
        (TaskAdmissionSpec("release_holes"), ordered),
    )
    replay = replay_admission(
        program,
        schedule,
        topology=ordered_topology,
    )
    reference = simulation_admission_from_replay(
        replay,
        program,
        schedule,
        device_capacity_bytes=20,
    )

    assert admitted.peak_allocated_bytes == 20
    assert admitted.decision_digest == replay.pool.decision_digest
    assert admitted.simulation_admission == reference
    task_deltas = {
        item.task_id: (item.start_bytes, item.completion_bytes)
        for item in admitted.simulation_admission.task_deltas
    }
    # All three allocations causally reuse ranges retired by release_holes.
    # The bytes remain physically charged across that task boundary, so the
    # successor adds no new physical capacity at task start.
    assert task_deltas["ordered_task"] == (0, -10)
    with pytest.raises(
        ValueError,
        match="physical admission requires explicit allocation steps",
    ):
        TaskAdmissionSpec(
            "ordered_task",
            workspace_extents=(4, 6),
            fresh_output_aliases=("out",),
        )


def test_admission_topology_uses_only_the_current_physical_schema() -> None:
    topology = _causal_topology()
    payload = topology.to_dict()

    assert payload["schema"] == "shadowspill.admission_topology/v3"
    assert AdmissionTopology.from_json(topology.to_json()) == topology

    payload["schema"] = "shadowspill.admission_topology/v2"
    with pytest.raises(ValueError, match="unsupported schema"):
        AdmissionTopology.from_dict(payload)


def test_pressurefit_repairs_fragmented_fetch_at_its_trigger_boundary() -> None:
    """Physical admission repairs one candidate without shrinking globally.

    The first repair delays ``d`` until ``before_d`` because releasing ``b``
    exposes only 32 bytes.  The second releases the adjacent 32-byte ``c``
    range at that boundary, producing the 64-byte destination and restoring
    ``c`` before its later consumer.
    """

    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    sizes = {"a": 64, "b": 32, "c": 32, "d": 64}
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=tuple(
            AliasGroupSpec(
                alias,
                "cuda_0",
                size,
                retain_spill_copy=alias in {"a", "c", "d"},
            )
            for alias, size in sizes.items()
        ),
        objects=tuple(
            ObjectSpec(alias, alias, 0, size) for alias, size in sizes.items()
        ),
        profiles=(TaskProfile("profile", 1_000, 0, "abi"),),
        tasks=(
            TaskSpec("release_b", compute, "profile", inputs=("b",)),
            TaskSpec(
                "before_d",
                compute,
                "profile",
                dependencies=("release_b",),
                inputs=("a", "c"),
            ),
            TaskSpec(
                "use_d",
                compute,
                "profile",
                dependencies=("before_d",),
                inputs=("d",),
            ),
            TaskSpec(
                "reuse_a_c",
                compute,
                "profile",
                dependencies=("use_d",),
                inputs=("a", "c"),
            ),
        ),
    )
    initial = (
        ResidencySpec("a", MemoryLocation.DEVICE),
        ResidencySpec("b", MemoryLocation.DEVICE),
        ResidencySpec("c", MemoryLocation.DEVICE),
        ResidencySpec("d", MemoryLocation.HOST),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=160,
        host_capacity_bytes=1_000,
        fetch_bandwidth_bytes_per_second=1_000_000,
        evict_bandwidth_bytes_per_second=1_000_000,
    )
    topology = AdmissionTopology(
        "cuda_0",
        160,
        160,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )

    with pytest.raises(PressureFitSearchExhaustedError) as exhausted:
        pressurefit(
            program,
            initial_residency=initial,
            config=config,
            admission=topology,
            options=PressureFitOptions(
                residency_strategies=("tight-stall",),
                prefetch_rules=("latest-safe",),
                evaluate_coalesced=False,
                max_repair_attempts=1,
                workers=1,
            ),
        )
    exhausted_candidates = tuple(
        item for item in exhausted.value.diagnostics if item.status == "exhausted"
    )
    assert len(exhausted_candidates) == 1
    assert exhausted_candidates[0].failure_kind == "repair_budget_exhausted"
    assert exhausted_candidates[0].repairs.total_attempts == 1

    result = pressurefit(
        program,
        initial_residency=initial,
        config=config,
        admission=topology,
        options=PressureFitOptions(
            residency_strategies=("tight-stall",),
            prefetch_rules=("latest-safe",),
            evaluate_coalesced=False,
            workers=1,
        ),
    )

    assert result.diagnostics.admission_refinements == ()
    diagnostic = result.diagnostics.recomputation_contexts[0].candidate_evaluations[0]
    assert diagnostic.repairs.total_attempts == 2
    assert diagnostic.repairs.admission_prefetch_delay_attempts == 1
    assert diagnostic.repairs.admission_pressure_boundary_attempts == 1
    assert diagnostic.work.admission_calls == 3
    assert diagnostic.work.simulation_calls == 1
    assert result.diagnostics.repairs == diagnostic.repairs
    assert tuple(
        (action.trigger_task_id, action.alias_group_id, action.kind)
        for action in result.schedule.actions
        if action.trigger_task_id == "before_d"
    ) == (
        ("before_d", "c", MemoryActionKind.RELEASE),
        ("before_d", "d", MemoryActionKind.PREFETCH),
    )
