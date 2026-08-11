from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    EntrypointSpec,
    ExecutionPlan,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    MutationSpec,
    ObjectSpec,
    PhysicalAdmission,
    PlanPrediction,
    Program,
    RecomputationGroup,
    RecomputationOption,
    RecomputationSelection,
    ResidencySpec,
    TaskSpec,
    ValidationError,
)

from ._examples import (
    SAVE_SELECTION,
    representative_plan,
    representative_program,
    representative_schedule,
)


def assert_invalid(path: str, call: Callable[[], object]) -> None:
    with pytest.raises(ValidationError) as caught:
        call()
    assert caught.value.path == path


def test_identifiers_and_integer_fields_are_strict() -> None:
    assert_invalid(
        "alias_group.alias_group_id",
        lambda: AliasGroupSpec("", "cuda_0", 1),
    )
    assert_invalid(
        "alias_group.device_id",
        lambda: AliasGroupSpec("storage", " cuda_0", 1),
    )
    assert_invalid(
        "alias_group.size_bytes",
        lambda: AliasGroupSpec("storage", "cuda_0", -1),
    )
    assert_invalid(
        "object.offset_bytes",
        lambda: ObjectSpec("object", "storage", True, 1),
    )


def test_program_collections_must_be_immutable_tuples() -> None:
    program = representative_program()

    assert_invalid(
        "program.devices",
        lambda: replace(program, devices=list(program.devices)),  # type: ignore[arg-type]
    )


def test_program_rejects_duplicate_and_unknown_identities() -> None:
    program = representative_program()
    duplicate = program.devices[0]
    assert_invalid(
        "program.devices[1]",
        lambda: replace(program, devices=(duplicate, duplicate)),
    )
    assert_invalid(
        "program.alias_groups[0].device_id",
        lambda: replace(
            program,
            alias_groups=(
                replace(program.alias_groups[0], device_id="missing"),
                *program.alias_groups[1:],
            ),
        ),
    )
    assert_invalid(
        "program.objects[0].alias_group_id",
        lambda: replace(
            program,
            objects=(
                replace(program.objects[0], alias_group_id="missing"),
                *program.objects[1:],
            ),
        ),
    )


def test_object_span_must_fit_its_alias_group() -> None:
    program = representative_program()

    assert_invalid(
        "program.objects[0]",
        lambda: replace(
            program,
            objects=(
                replace(program.objects[0], offset_bytes=32, size_bytes=64),
                *program.objects[1:],
            ),
        ),
    )


def test_task_graph_rejects_forward_and_missing_dependencies() -> None:
    program = representative_program()
    first = program.tasks[0]
    assert_invalid(
        "program.tasks[0].dependencies",
        lambda: replace(
            program,
            tasks=(replace(first, dependencies=("consume",)), *program.tasks[1:]),
        ),
    )
    assert_invalid(
        "program.tasks[3].dependencies",
        lambda: replace(
            program,
            tasks=(
                *program.tasks[:3],
                replace(program.tasks[3], dependencies=("backward_marker",)),
            ),
        ),
    )


def test_task_graph_rejects_unknown_records_and_invalid_mutations() -> None:
    program = representative_program()
    last = program.tasks[-1]
    assert_invalid(
        "program.tasks[3].profile_id",
        lambda: replace(
            program,
            tasks=(*program.tasks[:-1], replace(last, profile_id="missing")),
        ),
    )
    assert_invalid(
        "program.tasks[3].inputs",
        lambda: replace(
            program,
            tasks=(*program.tasks[:-1], replace(last, inputs=("missing",))),
        ),
    )
    assert_invalid(
        "program.tasks[3].mutations[0].object_id",
        lambda: replace(
            program,
            tasks=(
                *program.tasks[:-1],
                replace(
                    last,
                    mutations=(MutationSpec("weight"),),
                ),
            ),
        ),
    )
    assert_invalid(
        "task.mutations[1]",
        lambda: replace(
            last,
            inputs=("weight",),
            mutations=(MutationSpec("weight"), MutationSpec("weight")),
        ),
    )


def test_simultaneously_active_writers_are_rejected() -> None:
    program = representative_program()
    writer = TaskSpec(
        "extra_writer",
        program.tasks[0].resource,
        "forward_profile",
        outputs=("output",),
    )

    assert_invalid(
        "program.tasks[4].outputs",
        lambda: replace(program, tasks=(*program.tasks, writer)),
    )


def test_recomputation_groups_validate_membership_and_exclusivity() -> None:
    program = representative_program()
    unknown = RecomputationGroup(
        "bad_group",
        (RecomputationOption("bad_option", ("missing_task",)),),
    )
    assert_invalid(
        "program.recomputation_groups[1].options[0].active_task_ids",
        lambda: replace(
            program,
            recomputation_groups=(*program.recomputation_groups, unknown),
        ),
    )
    overlapping = RecomputationGroup(
        "other_group",
        (RecomputationOption("same_task", ("forward_save",)),),
    )
    assert_invalid(
        "program.recomputation_groups[1]",
        lambda: replace(
            program,
            recomputation_groups=(*program.recomputation_groups, overlapping),
        ),
    )


@pytest.mark.parametrize(
    "selections",
    [
        (),
        (
            RecomputationSelection("activation_tradeoff", "save"),
            RecomputationSelection("activation_tradeoff", "save"),
        ),
        (RecomputationSelection("missing_group", "save"),),
        (RecomputationSelection("activation_tradeoff", "missing_option"),),
    ],
)
def test_recomputation_selection_is_total_and_unique(
    selections: tuple[RecomputationSelection, ...],
) -> None:
    assert_invalid(
        "selections",
        lambda: representative_program().selected_tasks(selections),
    )


def test_schedule_rejects_unknown_and_out_of_order_actions() -> None:
    schedule = representative_schedule()
    assert_invalid(
        "schedule.actions[0].trigger_task_id",
        lambda: replace(
            schedule,
            actions=(
                replace(schedule.actions[0], trigger_task_id="missing"),
                *schedule.actions[1:],
            ),
        ).validate(representative_program(), SAVE_SELECTION),
    )
    assert_invalid(
        "schedule.actions[1]",
        lambda: replace(
            schedule,
            actions=(schedule.actions[2], schedule.actions[0]),
        ).validate(representative_program(), SAVE_SELECTION),
    )


@pytest.mark.parametrize(
    ("action", "path"),
    [
        (
            MemoryAction("forward_save", "output_storage", MemoryActionKind.RELEASE),
            "schedule.actions[0]",
        ),
        (
            MemoryAction("forward_save", "weight_storage", MemoryActionKind.PREFETCH),
            "schedule.actions[0]",
        ),
        (
            MemoryAction("forward_save", "output_storage", MemoryActionKind.OFFLOAD),
            "schedule.actions[0]",
        ),
    ],
)
def test_schedule_rejects_invalid_residency_transitions(
    action: MemoryAction,
    path: str,
) -> None:
    schedule = MemorySchedule(
        initial_residency=representative_schedule().initial_residency,
        actions=(action,),
    )

    assert_invalid(
        path,
        lambda: schedule.validate(representative_program(), SAVE_SELECTION),
    )


def test_schedule_checks_inputs_and_final_residency() -> None:
    schedule = representative_schedule()
    missing_weight = replace(
        schedule,
        initial_residency=(schedule.initial_residency[0],),
    )
    assert_invalid(
        "schedule.task[forward_save].inputs",
        lambda: missing_weight.validate(representative_program(), SAVE_SELECTION),
    )
    wrong_final = replace(
        schedule,
        final_residency=(ResidencySpec("output_storage", MemoryLocation.HOST),),
    )
    assert_invalid(
        "schedule.final_residency[0]",
        lambda: wrong_final.validate(representative_program(), SAVE_SELECTION),
    )


@pytest.mark.parametrize(
    ("changes", "path"),
    [
        ({"device_budget_bytes": -1}, "admission.device_budget_bytes"),
        ({"slab_bytes": 1000}, "admission.device_budget_bytes"),
        ({"workspace_reserve_bytes": 897}, "admission.workspace_reserve_bytes"),
        (
            {"predicted_fragmentation_bytes": 897},
            "admission.predicted_fragmentation_bytes",
        ),
        ({"host_reservation_bytes": 1025}, "admission.host_reservation_bytes"),
    ],
)
def test_physical_admission_rejects_budget_violations(
    changes: dict[str, int],
    path: str,
) -> None:
    assert_invalid(
        path,
        lambda: replace(representative_plan().admission, **changes),
    )


def test_execution_plan_requires_exact_entrypoint_bindings() -> None:
    plan = representative_plan()
    duplicate = plan.entrypoints[0]
    assert_invalid(
        "plan.entrypoints[1]",
        lambda: replace(plan, entrypoints=(duplicate, duplicate)),
    )
    assert_invalid(
        "plan.entrypoints",
        lambda: replace(plan, entrypoints=plan.entrypoints[:1]),
    )


@pytest.mark.parametrize(
    ("prediction", "path"),
    [
        (PlanPrediction(1025, 128, 38), "plan.prediction.device_peak_bytes"),
        (PlanPrediction(900, 1025, 38), "plan.prediction.host_peak_bytes"),
    ],
)
def test_execution_prediction_must_fit_public_budgets(
    prediction: PlanPrediction,
    path: str,
) -> None:
    assert_invalid(path, lambda: replace(representative_plan(), prediction=prediction))


@pytest.mark.parametrize(
    ("record", "value", "path"),
    [
        (Program, {}, "program.schema"),
        (Program, {"schema": "unknown"}, "program.schema"),
        (MemorySchedule, {"schema": "unknown"}, "schedule.schema"),
        (ExecutionPlan, {"schema": "unknown"}, "plan.schema"),
    ],
)
def test_wire_records_reject_missing_or_unknown_schemas(
    record: type[Program] | type[MemorySchedule] | type[ExecutionPlan],
    value: object,
    path: str,
) -> None:
    assert_invalid(path, lambda: record.from_dict(value))


def test_wire_records_report_nested_type_errors() -> None:
    value = representative_program().to_dict()
    value["devices"] = "not-an-array"

    assert_invalid("program.devices", lambda: Program.from_dict(value))


def test_admission_from_wire_rejects_boolean_integer() -> None:
    value = representative_plan().to_dict()
    admission = value["admission"]
    assert isinstance(admission, dict)
    admission["slab_bytes"] = True

    assert_invalid("plan.admission.slab_bytes", lambda: ExecutionPlan.from_dict(value))


def test_invalid_public_record_types_are_rejected() -> None:
    plan = representative_plan()
    assert_invalid(
        "plan.program",
        lambda: replace(plan, program=object()),  # type: ignore[arg-type]
    )
    assert_invalid(
        "plan.schedule",
        lambda: replace(plan, schedule=object()),  # type: ignore[arg-type]
    )


def test_memory_enums_reject_unknown_wire_values() -> None:
    schedule = representative_schedule().to_dict()
    initial = schedule["initial_residency"]
    assert isinstance(initial, list)
    assert isinstance(initial[0], dict)
    initial[0]["location"] = "elsewhere"
    assert_invalid(
        "schedule.initial_residency[0].location",
        lambda: MemorySchedule.from_dict(schedule),
    )


def test_entrypoint_fields_are_validated_from_wire() -> None:
    plan = representative_plan().to_dict()
    entrypoints = plan["entrypoints"]
    assert isinstance(entrypoints, list)
    assert isinstance(entrypoints[0], dict)
    entrypoints[0]["executor_id"] = 12

    assert_invalid(
        "plan.entrypoints[0].executor_id",
        lambda: ExecutionPlan.from_dict(plan),
    )


def test_constructor_rejects_invalid_entrypoint_and_prediction_values() -> None:
    assert_invalid(
        "entrypoint.task_id",
        lambda: EntrypointSpec("", "entry", "executor", "abi"),
    )
    assert_invalid(
        "prediction.makespan_ns",
        lambda: PlanPrediction(1, 1, -1),
    )


def test_physical_admission_wire_shape_is_stable() -> None:
    admission = representative_plan().admission
    restored = PhysicalAdmission.from_value(admission.to_dict(), "admission")

    assert restored == admission
