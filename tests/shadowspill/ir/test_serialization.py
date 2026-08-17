from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from shadowspill.ir import (
    ExecutionPlan,
    MemorySchedule,
    Program,
    index_execution_plan,
    index_memory_schedule,
    index_program,
)

from ._examples import (
    representative_plan,
    representative_program,
    representative_schedule,
)


def test_program_round_trip_is_byte_identical() -> None:
    program = representative_program()

    restored = Program.from_json(program.to_json())

    assert restored == program
    assert restored.to_json() == program.to_json()
    assert restored.digest == program.digest


def test_execution_plan_round_trip_is_byte_identical() -> None:
    plan = representative_plan()

    restored = ExecutionPlan.from_json(plan.to_json())

    assert restored == plan
    assert restored.to_json() == plan.to_json()
    assert restored.digest == plan.digest
    assert restored.scheduling_digest == plan.scheduling_digest


def test_embedded_schedule_has_the_same_wire_contract() -> None:
    plan = representative_plan()

    restored = MemorySchedule.from_json(plan.schedule.to_json())

    assert restored == plan.schedule
    assert restored.digest == plan.schedule.digest


def test_indexed_projection_has_stable_declared_order() -> None:
    indexed = index_program(representative_program())

    assert indexed.device_ids == ("cuda_0",)
    assert indexed.alias_group_ids == (
        "input_storage",
        "weight_storage",
        "activation_storage",
        "output_storage",
    )
    assert indexed.object_ids == ("input", "weight", "activation", "output")
    assert indexed.task_ids == (
        "forward_save",
        "backward_marker",
        "forward_recompute",
        "consume",
    )
    assert indexed.dependency_offsets == (0, 0, 1, 2, 5)
    assert indexed.dependencies == (0, 1, 0, 1, 2)
    assert indexed.input_offsets == (0, 2, 2, 4, 5)
    assert indexed.inputs == (0, 1, 0, 1, 2)
    assert indexed.output_offsets == (0, 1, 1, 2, 3)
    assert indexed.outputs == (2, 2, 3)
    assert indexed.alias_initial_version == (0, 0, 0, 0)
    assert indexed.alias_retain_spill_copy == (False, True, False, False)
    assert indexed.object_role == (0, 1, 3, 6)
    assert indexed.group_option_offsets == (0, 2)
    assert indexed.option_active_task_offsets == (0, 1, 2)
    assert indexed.option_active_tasks == (0, 2)
    assert indexed.option_retained_alias_offsets == (0, 1, 1)
    assert indexed.option_retained_aliases == (2,)


def test_schedule_projection_preserves_ordered_actions() -> None:
    indexed = index_memory_schedule(
        representative_program(),
        representative_schedule(),
    )

    assert indexed.initial_alias_groups == (0, 1)
    assert indexed.initial_locations == (0, 0)
    assert indexed.action_trigger_tasks == (0, 1, 3)
    assert indexed.action_alias_groups == (2, 2, 2)
    assert indexed.action_kinds == (1, 2, 0)
    assert indexed.final_alias_groups == (3,)
    assert indexed.final_locations == (0,)


def test_execution_projection_contains_resolved_admission() -> None:
    indexed = index_execution_plan(representative_plan())

    assert indexed.selection_groups == (0,)
    assert indexed.selection_options == (0,)
    assert indexed.entrypoint_tasks == (0, 3)
    assert indexed.entrypoint_ids == ("forward_entrypoint", "consume_entrypoint")
    assert indexed.device_budget_bytes == 1024
    assert indexed.slab_bytes == 896
    assert indexed.workspace_reserve_bytes == 128
    assert indexed.predicted_device_peak_bytes == 900
    assert indexed.predicted_makespan_ns == 38


def test_canonical_ir_matches_frozen_identity_artifact() -> None:
    plan = representative_plan()
    indexed = index_execution_plan(plan)
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads((root / "tests/fixtures/ir/canonical_ir_v2.json").read_text())

    assert artifact["program_digest"] == plan.program.digest
    assert artifact["schedule_digest"] == plan.schedule.digest
    assert artifact["execution_plan_digest"] == plan.digest
    assert artifact["scheduling_digest"] == plan.scheduling_digest
    assert artifact["device_ids"] == list(indexed.program.device_ids)
    assert artifact["alias_group_ids"] == list(indexed.program.alias_group_ids)
    assert artifact["object_ids"] == list(indexed.program.object_ids)
    assert artifact["task_ids"] == list(indexed.program.task_ids)
    assert artifact["selected_option_indices"] == list(indexed.selection_options)
    assert artifact["action_trigger_task_indices"] == list(
        indexed.schedule.action_trigger_tasks
    )
    assert artifact["action_alias_group_indices"] == list(
        indexed.schedule.action_alias_groups
    )
    assert artifact["action_kind_codes"] == list(indexed.schedule.action_kinds)


def test_ir_import_does_not_load_framework_or_backends() -> None:
    for name in tuple(sys.modules):
        if name == "shadowspill.ir" or name.startswith("shadowspill.ir."):
            del sys.modules[name]
    before = set(sys.modules)

    importlib.import_module("shadowspill.ir")

    imported = set(sys.modules) - before
    assert "torch" not in imported
    assert "mlops" not in imported
    assert not any(name.startswith("shadowspill.runtime") for name in imported)
