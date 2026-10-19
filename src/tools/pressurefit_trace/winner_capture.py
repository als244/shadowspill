"""Save an admitted quickstart winner without changing planning or execution.

The observer is opt-in and writes outside timed steps. Files contain no tensors
or compiled callables. Distributed planning's fixed-resolution search input is
retained separately from the restored program used for physical admission.
"""

from __future__ import annotations

import functools
import json
import traceback
from dataclasses import replace
from pathlib import Path


def annotated_selection(
    result,
    facts,
    admission,
    *,
    execution_bytes,
    spill_bytes,
    attempts=(),
    from_store=False,
):
    from shadowspill.ir import shared_residency_footprint
    from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
    from shadowspill.planner.program_inputs import (
        MemoryBudgets,
        ShadowSpillPlanningProblem,
    )

    shared = shared_residency_footprint(result.program).for_device(facts.device_id)
    config = replace(
        result.simulation_config,
        devices=tuple(
            replace(device, capacity_bytes=facts.object_capacity_bytes + shared)
            if device.device_id == facts.device_id
            else device
            for device in result.simulation_config.devices
        ),
    )
    problem = ShadowSpillPlanningProblem(
        role="step",
        program=result.program,
        initial_residency=result.initial_residency,
        final_residency=result.final_residency,
        simulation_config=config,
        admission_facts=facts,
        source_execution_budget_bytes=execution_bytes,
        maximum_execution_budget_bytes=execution_bytes,
        maximum_spill_budget_bytes=spill_bytes,
        fixed_execution_bytes=execution_bytes - facts.pool_capacity_bytes - shared,
        object_reserve_bytes=facts.pool_capacity_bytes - facts.object_capacity_bytes,
        dynamic_scratch_reserve_bytes=admission.layout.scratch_reserve_bytes,
    )
    physical = replace(
        result,
        simulation=admission.simulation,
        diagnostics=result.diagnostics.replace_selected_makespan(
            admission.simulation.makespan_ns
        ),
    )
    return AnnotatedProgramPlan(
        program=problem,
        memory_budgets=MemoryBudgets(execution_bytes, spill_bytes),
        transfer_bandwidths=problem.transfer_bandwidths,
        result=physical,
        effective_facts=facts,
        fixed_layout=admission.layout,
        simulation_admission=admission.simulator_input,
        simulation=admission.simulation,
        attempts=attempts,
        plan_from_store=from_store,
        wall_time_ns=sum(
            a.search_wall_time_ns + a.physical_admission_wall_time_ns for a in attempts
        ),
    )


def install(directory):
    """Observe subsequently admitted training plans; return an uninstall callback.

    One process owns one directory (use rank-specific directories). Export errors
    are recorded and do not discard an otherwise valid benchmark measurement.
    A downstream collector must require bundle.json before using the evidence.
    """
    from shadowspill.planner.serialization.simulation_result import (
        simulation_result_value,
    )
    from shadowspill.pytorch.diagnostics.semantics import training_semantic_labels
    from shadowspill.pytorch.distributed import _selection
    from shadowspill.pytorch.planning.training import admit

    directory = Path(directory)
    original_restore = _selection.restore_result
    original_report = admit.training_plan_report
    last = []

    @functools.wraps(original_restore)
    def restored(local, *args, **kwargs):
        answer = original_restore(local, *args, **kwargs)
        last[:] = [(answer.program.digest, answer.schedule.digest, local)]
        return answer

    @functools.wraps(original_report)
    def reported(model, captured, profiled, programs, selection, admitted, **kwargs):
        report = original_report(
            model, captured, profiled, programs, selection, admitted, **kwargs
        )
        target = directory / f"{report.execution_budget_bytes}bytes"
        target.mkdir(parents=True, exist_ok=True)
        try:
            plan = annotated_selection(
                admitted.result,
                selection.facts,
                selection.admission,
                execution_bytes=report.execution_budget_bytes,
                spill_bytes=report.spill_budget_bytes,
                attempts=selection.attempts,
                from_store=selection.from_store,
            )
            # Decoding checks the program, schedule, facts, layout and makespan.
            from shadowspill.planner.annotated_plan import AnnotatedProgramPlan

            encoded = plan.to_json()
            AnnotatedProgramPlan.from_json(encoded)
            (target / "plan.json").write_text(encoded + "\n")
            semantics = training_semantic_labels(programs.lowered)
            semantics["program_digest"] = plan.result.program.digest
            (target / "semantics.json").write_text(json.dumps(semantics) + "\n")
            local = selection
            if last and last[0][:2] == (
                plan.result.program.digest,
                plan.result.schedule.digest,
            ):
                local = last[0][2]
            trace_plan = annotated_selection(
                local.result,
                local.facts,
                local.admission,
                execution_bytes=report.execution_budget_bytes,
                spill_bytes=report.spill_budget_bytes,
                attempts=local.attempts,
                from_store=local.from_store,
            )
            (target / "search-plan.json").write_text(trace_plan.to_json() + "\n")
            (target / "search-logical.json").write_text(
                json.dumps(
                    {
                        "simulation": simulation_result_value(local.result.simulation),
                        "diagnostics": local.result.diagnostics.to_dict(),
                    }
                )
                + "\n"
            )
            record = {
                "schema": "pressurefit_winner_bundle/v1",
                "program_digest": plan.result.program.digest,
                "schedule_digest": plan.result.schedule.digest,
                "layout_digest": plan.fixed_layout.digest,
                "capture_identity": report.capture_identity,
                "execution_budget_bytes": report.execution_budget_bytes,
                "spill_budget_bytes": report.spill_budget_bytes,
                "ordering": report.data_ordering.to_dict(),
                "ordering_label": report.data_ordering.label,
                "search_program_digest": trace_plan.result.program.digest,
                "search_candidate_id": local.result.diagnostics.selected_candidate_id,
                "search_selection_id": local.result.diagnostics.selected_selection_id,
                "selections": [c.to_dict() for c in plan.result.selections],
                "summary": report.summary.as_dict(),
                "status": "admitted",
            }
            temporary = target / "bundle.json.tmp"
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(target / "bundle.json")
        except Exception:
            (target / "export-error.txt").write_text(traceback.format_exc())
            print(f"Winner export failed: {target / 'export-error.txt'}", flush=True)
        finally:
            last.clear()
        return report

    _selection.restore_result = restored
    admit.training_plan_report = reported

    def uninstall():
        _selection.restore_result = original_restore
        admit.training_plan_report = original_report
        last.clear()

    return uninstall
