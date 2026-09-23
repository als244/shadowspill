"""The qualification artifact: everything one planned case can be judged on.

It is written whether the case passed or failed, because a failure is only
actionable with the numbers beside it: which tensors disagreed and by how
much, what the plan predicted, what the pools actually held, and which caches
served the run.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from shadowspill.schema import artifact_schema
from shadowspill.store import StorePolicy

from ..planning_phases import planning_breakdown
from .compare import Comparison
from .measures import (
    failure_tensor_values,
    optimizer_steps,
    recomputation_savings_bytes,
)
from .metrics import state_digest
from .references import reference_inputs_path
from .request import REFERENCE_EXECUTION, PlannedRequest
from .run import PlannedRun
from .tolerances import (
    LOSS_ABSOLUTE_TOLERANCE,
    LOSS_RELATIVE_TOLERANCE,
    MAXIMUM_RELATIVE_L2,
    MAXIMUM_RELATIVE_L2_OPTIMIZER,
    MINIMUM_COSINE,
    MINIMUM_SIGN_AGREEMENT,
    failures_by_state,
)
from .verdict import transfer_pressure_gate_passed


def qualification_artifact(
    request: PlannedRequest,
    run: PlannedRun,
    comparison: Comparison,
) -> dict[str, Any]:
    """Assemble the case artifact: diagnostics, plan records, state digests."""

    case_name = f"{request.case.model_implementation}/{request.case.family}"
    print(
        f"shadowspill {case_name} assembling the case "
        "artifact: diagnostics, plan records, and the state digests",
        flush=True,
    )
    qualification_result: dict[str, Any] = {
        **_case_identity(request, run, comparison),
        **_timings(run, comparison),
        **_agreement(run, comparison),
        **_checkpoint_replay(run, comparison),
        **_plan_records(run),
        **_pools(run),
        **_caches(request, run),
        **_digests(run, comparison),
    }
    if request.detailed_artifacts:
        qualification_result["plan_diagnostics"] = run.report.diagnostics.as_dict()
        qualification_result["planned_step_diagnostics"] = run.step_diagnostics
    _record_gates(qualification_result, request, run)
    return qualification_result


def _record_gates(
    qualification_result: dict[str, Any],
    request: PlannedRequest,
    run: PlannedRun,
) -> None:
    """Judge the assembled evidence, naming the kind of failure each gate reports.

    "The cell failed" is not actionable: disagreeing with a reference recorded
    on other hardware, failing to reproduce its own replay, and exceeding a
    budget are different problems with different owners.
    """

    qualification_result["reference_bitwise_equal"] = bool(
        qualification_result["reference_state_digest"]
        == qualification_result["planned_state_digest"]
    )
    # A store mode that disables reads must actually have served nothing. The
    # policy the mode implies is the authority on whether reads were allowed,
    # so there is no second flag to disagree with it; a tree that was allowed
    # to read is not asked to prove anything.
    build_reads = StorePolicy.for_mode(request.build_store_mode).read_enabled
    plan_reads = StorePolicy.for_mode(request.plan_store_mode).read_enabled
    qualification_result["store_modes_honoured"] = bool(
        (
            build_reads
            or (
                qualification_result["profile_cache_hits"] == 0
                and qualification_result["aot_graph_pair_cache_misses"]
                == qualification_result["aot_unique_stage_contracts"]
            )
        )
        and (plan_reads or qualification_result["planned_program_cache_hits"] == 0)
    )
    qualification_result["graph_pair_selection_required"] = False
    transfer_pressure_passed = transfer_pressure_gate_passed(
        required=request.require_pressure,
        evicted_bytes=run.report.transfer_bytes_evicted,
        fetched_bytes=run.report.transfer_bytes_fetched,
    )
    qualification_result["transfer_pressure_gate_passed"] = transfer_pressure_passed
    qualification_result["pressure_gate_passed"] = transfer_pressure_passed


def _case_identity(
    request: PlannedRequest, run: PlannedRun, comparison: Comparison
) -> dict[str, Any]:
    """Who the case is, what it was asked to do, and what it is judged against."""

    return {
        "schema": artifact_schema("numerical_qualification"),
        "reference_execution": REFERENCE_EXECUTION,
        "family": request.case.family,
        "model_implementation": request.case.model_implementation,
        "case_identity": request.case.identity(),
        "reference_artifact": {
            "path": str(request.reference_path.resolve()),
            "schema": comparison.reference["schema"],
            "inputs_path": str(reference_inputs_path(request.reference_path).resolve()),
            "microbatch_digest": run.requested_input_digest,
        },
        "steps": request.case.steps,
        "checkpoint_step": request.checkpoint_step,
        "require_pressure": request.require_pressure,
        "case_request": {
            "model_name": request.case.family,
            "model_implementation": request.case.model_implementation,
            "seed": request.case.seed,
            "model_config": request.case.model_config,
            "data_geometry": request.case.data_geometry,
            "case_factory": request.case.case_factory,
            "case_options": request.case.case_options,
            "optimizer_ordering": request.case.optimizer_ordering,
            "data_ordering": request.case.data_ordering,
            "profiling_metadata": run.workload_metadata,
        },
        "planning_cache_request": {
            "directory": (
                None
                if request.artifact_store is None
                else str(request.artifact_store.resolve())
            ),
            "build_store_mode": request.build_store_mode,
            "plan_store_mode": request.plan_store_mode,
            "export_bypass_key": request.export_bypass_key,
        },
        "device_budget_bytes": request.device_budget,
        "tolerances": {
            "loss_rtol": LOSS_RELATIVE_TOLERANCE,
            "loss_atol": LOSS_ABSOLUTE_TOLERANCE,
            "minimum_cosine": MINIMUM_COSINE,
            "maximum_relative_l2": MAXIMUM_RELATIVE_L2,
            "maximum_relative_l2_optimizer": MAXIMUM_RELATIVE_L2_OPTIMIZER,
            "minimum_sign_agreement": MINIMUM_SIGN_AGREEMENT,
        },
        "artifact_detail": "detailed" if request.detailed_artifacts else "compact",
    }


def _timings(run: PlannedRun, comparison: Comparison) -> dict[str, Any]:
    """How long planning took, and the step times both arms measured."""

    return {
        "planning_seconds": run.planning_seconds,
        "phase_seconds": run.phase_seconds,
        "planning_breakdown_seconds": planning_breakdown(
            run.phase_seconds, planning_seconds=run.planning_seconds
        ),
        "search_seconds": run.phase_seconds.get("search", 0.0),
        "planned_step_seconds": run.timings,
        "planned_compute_seconds": run.compute_timings,
        "planned_step_summaries": run.step_summaries,
        "reference_step_seconds": comparison.reference["step_seconds"],
        "reference_compute_seconds": comparison.reference.get(
            "compute_step_seconds", []
        ),
        "reference_execution_timings": comparison.reference.get(
            "execution_timings", []
        ),
    }


def _agreement(run: PlannedRun, comparison: Comparison) -> dict[str, Any]:
    """Where the two arms disagreed: losses, tensor metrics, and structure."""

    return {
        "planned_losses": run.losses,
        "reference_losses": comparison.reference["losses"],
        "worst_loss_relative": comparison.worst_loss_relative,
        "loss_failures": comparison.loss_failures,
        "minimum_cosine": min(
            (item.cosine for item in comparison.tensor_results.values()), default=1.0
        ),
        "maximum_relative_l2": max(
            (item.relative_l2 for item in comparison.tensor_results.values()),
            default=0.0,
        ),
        "minimum_sign_agreement": min(
            (item.sign_agreement for item in comparison.tensor_results.values()),
            default=1.0,
        ),
        "metric_failure_keys": comparison.metric_failures,
        "metric_failures_by_state": failures_by_state(comparison.metric_failures),
        "exact_failures_by_state": failures_by_state(comparison.exact_failures),
        "structure_failures_by_state": failures_by_state(comparison.structure_failures),
        "metric_failures": {
            name: asdict(comparison.tensor_results[name])
            for name in comparison.metric_failures
        },
        "metric_failure_values": failure_tensor_values(
            comparison.metric_failures,
            {
                "model": comparison.reference["model"],
                "optimizer": comparison.reference["optimizer"],
            },
            {
                "model": run.final_state["model"],
                "optimizer": run.final_state["optimizer"],
            },
        ),
        "exact_failures": comparison.exact_failures,
        "structure_failures": comparison.structure_failures,
    }


def _checkpoint_replay(run: PlannedRun, comparison: Comparison) -> dict[str, Any]:
    """Whether resuming from the checkpoint reproduced the uninterrupted run."""

    return {
        "checkpoint_replay_bitwise": run.uninterrupted_digest == run.replay_digest
        and run.expected_replay == run.replay_losses,
        "checkpoint_replay_within_tolerance": not comparison.replay_exact_failures
        and not comparison.replay_metric_failures,
        "checkpoint_replay_exact_failures": comparison.replay_exact_failures,
        "checkpoint_replay_metric_failure_keys": comparison.replay_metric_failures,
        "checkpoint_replay_metric_failures_by_state": failures_by_state(
            comparison.replay_metric_failures
        ),
        "checkpoint_replay_metric_failures": {
            name: asdict(comparison.replay_results[name])
            for name in comparison.replay_metric_failures
        },
        "checkpoint_replay_minimum_cosine": min(
            (item.cosine for item in comparison.replay_results.values()), default=1.0
        ),
        "checkpoint_replay_maximum_relative_l2": max(
            (item.relative_l2 for item in comparison.replay_results.values()),
            default=0.0,
        ),
        "checkpoint_replay_maximum_absolute_error": max(
            (
                item.maximum_absolute_error
                for item in comparison.replay_results.values()
            ),
            default=0.0,
        ),
        "checkpoint_replay_tensors_compared": len(comparison.replay_results),
        "checkpoint_replay_tensors_differing": sum(
            1
            for item in comparison.replay_results.values()
            if item.difference_norm > 0.0
        ),
        "checkpoint_steps": optimizer_steps(run.checkpoint),
        "uninterrupted_steps": optimizer_steps(run.uninterrupted_state),
        "replay_steps": optimizer_steps(run.final_state),
    }


def _plan_records(run: PlannedRun) -> dict[str, Any]:
    """What the plan chose and what it predicted the step would cost."""

    selections = tuple(
        (item.group_id, item.option_id) for item in run.report.execution_plan.selections
    )
    available_recomputation_savings, selected_recomputation_savings = (
        recomputation_savings_bytes(
            run.report.execution_plan.program.task_alternative_groups,
            run.report.execution_plan.selections,
            {
                group.alias_group_id: group.size_bytes
                for group in run.report.execution_plan.program.alias_groups
            },
        )
    )
    return {
        "transfer_bytes_evicted": run.report.transfer_bytes_evicted,
        "transfer_bytes_fetched": run.report.transfer_bytes_fetched,
        "selected_recomputation": any(
            option != "save" for _group, option in selections
        ),
        "recomputation_memory_saving_available": bool(available_recomputation_savings),
        "maximum_recomputation_savings_bytes": available_recomputation_savings,
        "selected_recomputation_savings_bytes": selected_recomputation_savings,
        "task_alternative_group_count": len(selections),
        "task_count": len(
            run.report.execution_plan.program.selected_tasks(
                run.report.execution_plan.selections
            )
        ),
        "action_count": len(run.report.transfer_actions),
        "predicted_makespan_seconds": run.report.predicted_makespan_ns / 1e9,
        "predicted_device_peak_bytes": run.report.predicted_device_peak_bytes,
        "predicted_spill_peak_bytes": run.report.predicted_spill_peak_bytes,
        "predicted_fragmentation_bytes": (
            run.report.execution_plan.admission.predicted_fragmentation_bytes
        ),
        "fixed_slab_bytes": run.report.fixed_slab_bytes,
    }


def _pools(run: PlannedRun) -> dict[str, Any]:
    """What the pools and their record tables actually held during the run."""

    return {
        "physical_budget_statuses": run.physical_statuses,
        "physical_budget_sealed": bool(run.runtime_statistics.physical_budget_sealed),
        "peak_process_physical_bytes": int(
            run.runtime_statistics.peak_process_physical_bytes
        ),
        "observed_external_high_water_bytes": int(
            run.runtime_statistics.observed_external_high_water_bytes
        ),
        "execution_pool_bytes": int(
            run.runtime_statistics.allocator_pool.capacity_bytes
        ),
        "slab_peak_allocated_bytes": int(
            run.runtime_statistics.allocator_pool.peak_allocated_bytes
        ),
        "spill_pool_bytes": int(run.spill_statistics.capacity_bytes),
        "spill_peak_allocated_bytes": int(run.spill_statistics.peak_allocated_bytes),
        "callback_failures": int(run.runtime_statistics.callback_failures),
        "pointer_lookup_failures": int(run.runtime_statistics.pointer_lookup_failures),
        "allocation_event_overflow": bool(
            run.runtime_statistics.runtime.allocation_event_overflow
        ),
        "neutral_event_lease_capacity": int(
            run.runtime_statistics.runtime.event_lease_capacity
        ),
        "neutral_event_lease_peak_in_use": int(
            run.runtime_statistics.runtime.event_lease_peak_in_use
        ),
        "neutral_event_lease_growth_rejections": int(
            run.runtime_statistics.runtime.event_lease_growth_rejections
        ),
        "retirement_record_capacity": int(
            run.runtime_statistics.runtime.retirement_record_capacity
        ),
        "retirement_record_peak_in_use": int(
            run.runtime_statistics.runtime.retirement_record_peak_in_use
        ),
        "retirement_record_growth_rejections": int(
            run.runtime_statistics.runtime.retirement_record_growth_rejections
        ),
        "memory_lease_record_capacity": int(
            run.runtime_statistics.allocator_pool.memory_lease_record_capacity
        ),
        "memory_lease_record_peak_in_use": int(
            run.runtime_statistics.allocator_pool.memory_lease_record_peak_in_use
        ),
        "memory_lease_record_growth_rejections": int(
            run.runtime_statistics.allocator_pool.memory_lease_record_growth_rejections
        ),
        "lease_use_record_capacity": int(
            run.runtime_statistics.allocator_pool.lease_use_record_capacity
        ),
        "lease_use_record_peak_in_use": int(
            run.runtime_statistics.allocator_pool.lease_use_record_peak_in_use
        ),
        "lease_use_record_growth_rejections": int(
            run.runtime_statistics.allocator_pool.lease_use_record_growth_rejections
        ),
        "backend_device_allocations": int(
            run.runtime_statistics.backend.device_allocations
        ),
        "steady_state_backend_device_allocations": int(
            run.runtime_statistics.backend.device_allocations
            - run.execution_baseline.backend.device_allocations
        ),
        "steady_state_pinned_host_registrations": int(
            run.runtime_statistics.backend.pinned_host_registrations
            - run.execution_baseline.backend.pinned_host_registrations
        ),
        "event_pool_capacity": int(run.runtime_statistics.runtime.event_lease_capacity),
        "event_pool_peak_in_use": int(
            run.runtime_statistics.runtime.event_lease_peak_in_use
        ),
        "event_pool_driver_creates": int(
            run.runtime_statistics.runtime.event_lease_driver_creates
        ),
        "steady_state_event_pool_driver_creates": int(
            run.runtime_statistics.runtime.event_lease_driver_creates
            - run.execution_baseline.runtime.event_lease_driver_creates
        ),
        "event_pool_growth_rejections": int(
            run.runtime_statistics.runtime.event_lease_growth_rejections
        ),
        "event_pool_sealed": bool(run.runtime_statistics.runtime.event_lease_sealed),
    }


def _caches(request: PlannedRequest, run: PlannedRun) -> dict[str, Any]:
    """Which caches served the run, and the plan records it wrote."""

    return {
        "profile_cache_hits": run.report.profile_cache_hits,
        "profile_cache_misses": run.report.profile_cache_misses,
        "profile_unique_keys": run.report.profile_unique_keys,
        "captured_stage_count": run.report.captured_stage_count,
        "aot_unique_stage_contracts": run.report.aot_unique_stage_contracts,
        "aot_graph_pair_cache_hits": run.report.aot_graph_pair_cache_hits,
        "aot_graph_pair_cache_misses": run.report.aot_graph_pair_cache_misses,
        "planned_program_cache_hits": run.report.planned_program_cache_hits,
        "planned_program_cache_misses": run.report.planned_program_cache_misses,
        "build_store_mode": request.build_store_mode,
        "plan_store_mode": request.plan_store_mode,
        "plan_records": run.plan_records,
        "plan_report_artifact": run.plan_report_artifact,
    }


def _digests(run: PlannedRun, comparison: Comparison) -> dict[str, Any]:
    """One digest per arm, over the model and optimizer state it ended with."""

    return {
        "reference_state_digest": state_digest(
            {
                "model": comparison.reference["model"],
                "optimizer": comparison.reference["optimizer"],
            }
        ),
        "planned_state_digest": state_digest(
            {
                "model": run.final_state["model"],
                "optimizer": run.final_state["optimizer"],
            }
        ),
    }
