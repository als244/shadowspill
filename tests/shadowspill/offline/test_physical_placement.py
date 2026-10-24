"""Fixed executable placement need not fit a different, first-fit allocator."""

from dataclasses import replace

import pytest

from shadowspill import MemoryBudgets, apply_profiles, plan_program
from shadowspill.artifacts._legacy import split_costed_program
from shadowspill.errors import PlanInfeasibleError
from shadowspill.ir import (
    AliasGroupSpec,
    MemoryLocation,
    ObjectSpec,
    ResidencySpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planning.admission.program import program_admission
from shadowspill.planning.costs import simulation_inputs
from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.task.layout import reconcile_compiled_task_layout
from shadowspill.task.profiles import TaskMeasurement
from shadowspill.task.storage import TaskStorageContract
from tests.shadowspill.planner._examples import COMPUTE, DEVICE

from ._example import communication, options


def test_fixed_plan_reuses_space_that_fragments_dynamic_replay():
    # Initial first-fit order is A/B/C. A and C die together, leaving two
    # disjoint 256-byte holes around B. A fixed layout can place B at the end
    # and reuse A/C's adjacent slots for the later 512-byte workspace.
    captured = ShadowSpillProgram(
        devices=(DEVICE,),
        alias_groups=tuple(AliasGroupSpec(a, "cuda_0", 256) for a in "abc"),
        objects=tuple(ObjectSpec(a, a, 0, 256) for a in "abc"),
        profiles=(
            TaskProfile("first", 10, 0, "first"),
            TaskProfile("last", 10, 512, "last"),
        ),
        tasks=(
            TaskSpec("first", COMPUTE, "first", inputs=("a", "b", "c")),
            TaskSpec("last", COMPUTE, "last", inputs=("b",), dependencies=("first",)),
        ),
    )
    program, profiles = split_costed_program(
        captured,
        initial_residency=tuple(ResidencySpec(a, MemoryLocation.DEVICE) for a in "abc"),
        fixed_device_bytes=0,
    )
    storage = TaskStorageContract.build((), (), ())
    for key, record in profiles.tasks.items():
        size = record.workspace_bytes
        trace = (
            ()
            if not size
            else tuple(
                TaskAllocationEvent(0, operation, size, size)
                for operation in (
                    TaskAllocationOperation.ALLOCATE,
                    TaskAllocationOperation.FREE,
                )
            )
        )
        measurement = TaskMeasurement(
            10, size, size, (), (10,), "fixture", allocation_trace=trace
        )
        profiles.tasks[key] = replace(
            record,
            metadata={
                "duration_kind": "static",
                "memory_kind": "static",
                "allocation_trace": [event.to_dict() for event in trace],
                "allocation_contract": TaskAllocationContract.capture(
                    trace, storage
                ).to_dict(),
                "compiled_layout": reconcile_compiled_task_layout(
                    storage, measurement
                ).identity(),
                "output_input_bindings": [],
                "off_device_output_leaves": [],
            },
        )
    profiles.metadata["memory_kind"] = "static"
    program = replace(
        program,
        executables={key: b"fixture: not executed" for key in profiles.tasks},
        bindings={
            "manifests": {
                key: {"storage_contract": storage.to_dict(), "root_allocations": []}
                for key in profiles.tasks
            },
            "entrypoints": [
                {"task_id": key, "input_slots": [], "output_slots": []}
                for key in profiles.tasks
            ],
        },
    )
    profiled = apply_profiles(program, profiles)
    budgets = MemoryBudgets(768, 768)
    costed, config = simulation_inputs(profiled, budgets, communication())
    facts = program_admission(profiled, costed, 768)
    search = options()
    with pytest.raises(PlanInfeasibleError):
        search.resolved_algorithm(
            costed,
            initial_residency=program.initial_residency,
            final_residency=(),
            config=config,
            generic=search.generic,
            workers=1,
            admission=facts,
            placement=facts,
        )

    plan = plan_program(
        profiled,
        memory_budgets=budgets,
        communication_info=communication(),
        search_options=search,
    )
    assert plan.memory_layout is not None
    assert plan.memory_layout.required_bytes == 768
    assert plan.timeline.makespan_ns == 20
