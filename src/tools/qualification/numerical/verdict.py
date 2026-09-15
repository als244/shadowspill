"""The checks one planned case must satisfy, and the failures it reports.

Every check names the kind of failure it reports, because "the cell failed"
is not actionable: disagreeing with a reference recorded on other hardware,
failing to reproduce its own replay, and exceeding a budget are different
problems with different owners.
"""

from __future__ import annotations

from typing import Any

from .compare import Comparison
from .request import PlannedRequest
from .run import PlannedRun
from .tolerances import SPILL_BUDGET, state_split


def transfer_pressure_gate_passed(
    *, required: bool, evicted_bytes: int, fetched_bytes: int
) -> bool:
    """Require real bidirectional movement, never a planner policy choice."""

    return bool(not required or (evicted_bytes > 0 and fetched_bytes > 0))


#: One check: the kind of failure it reports, whether it held, and what to say.
_Check = tuple[str, object, str]


def _reference_checks(
    result: dict[str, Any], comparison: Comparison
) -> tuple[_Check, ...]:
    """Whether the planned arm answered what the reference arm answered."""

    structure = comparison.structure_failures
    return (
        (
            "reference",
            not comparison.loss_failures,
            f"{len(comparison.loss_failures)} losses differ",
        ),
        # First, because it decides whether the rest means anything. States
        # that do not have the same shape are not two answers to one question,
        # so the tolerance numbers below are taken over tensors that do not
        # correspond and describe nothing.
        (
            "reference",
            not structure,
            f"{len(structure)} tensors do not match the reference's structure "
            f"[{state_split(structure)}]: the reference records a different "
            "shape, so the comparison is not a numerical one -- first is "
            f"{structure[0] if structure else ''}",
        ),
        (
            "reference",
            not comparison.metric_failures,
            f"{len(comparison.metric_failures)} tensors outside tolerance "
            f"[{state_split(comparison.metric_failures)}] "
            f"(min cosine {result['minimum_cosine']:.6f}, "
            f"max relative l2 "
            f"{result['maximum_relative_l2']:.6f})",
        ),
        (
            "reference",
            not comparison.exact_failures,
            f"{len(comparison.exact_failures)} tensors differ that must match exactly "
            f"[{state_split(comparison.exact_failures)}]",
        ),
    )


def _replay_checks(
    result: dict[str, Any], comparison: Comparison
) -> tuple[_Check, ...]:
    """Whether resuming from the checkpoint reproduced the run it interrupted."""

    return (
        (
            "replay",
            not comparison.replay_structure_failures,
            f"{len(comparison.replay_structure_failures)} tensors do not match the "
            "replay's structure "
            f"[{state_split(comparison.replay_structure_failures)}]",
        ),
        (
            "replay",
            result["checkpoint_replay_within_tolerance"],
            f"{result['checkpoint_replay_tensors_differing']} of "
            f"{result['checkpoint_replay_tensors_compared']} "
            "tensors differ from the checkpoint replay "
            f"[{state_split(comparison.replay_metric_failures)}] (min cosine "
            f"{result['checkpoint_replay_minimum_cosine']:.6f})",
        ),
    )


def _budget_checks(
    result: dict[str, Any], request: PlannedRequest, run: PlannedRun
) -> tuple[_Check, ...]:
    """Whether every pool held what it was declared to hold, and no more."""

    return (
        (
            "budget",
            run.report.predicted_device_peak_bytes <= request.device_budget,
            f"predicted device peak {run.report.predicted_device_peak_bytes} "
            f"over budget {request.device_budget}",
        ),
        (
            "budget",
            not any(run.physical_statuses),
            f"physical statuses {run.physical_statuses}",
        ),
        ("budget", result["physical_budget_sealed"], "budget not sealed"),
        (
            "budget",
            result["peak_process_physical_bytes"] <= request.device_budget,
            f"process peak {result['peak_process_physical_bytes']} "
            f"over budget {request.device_budget}",
        ),
        (
            "budget",
            result["slab_peak_allocated_bytes"] <= result["execution_pool_bytes"],
            "slab peak over the execution pool",
        ),
        (
            "budget",
            result["spill_peak_allocated_bytes"]
            <= result["spill_pool_bytes"]
            <= SPILL_BUDGET,
            "spill peak over the pool, or the pool over its budget",
        ),
    )


def _runtime_checks(result: dict[str, Any]) -> tuple[_Check, ...]:
    """Whether the runtime reached steady state and stayed there."""

    return (
        ("runtime", result["callback_failures"] == 0, "callback failures"),
        ("runtime", result["pointer_lookup_failures"] == 0, "pointer lookup failures"),
        (
            "runtime",
            not result["allocation_event_overflow"],
            "allocation event overflow",
        ),
        (
            "runtime",
            result["backend_device_allocations"] == 1,
            f"{result['backend_device_allocations']} backend device "
            "allocations, expected one",
        ),
        (
            "runtime",
            result["steady_state_backend_device_allocations"] == 0,
            "device allocations in steady state",
        ),
        (
            "runtime",
            result["steady_state_pinned_host_registrations"] == 0,
            "pinned host registrations in steady state",
        ),
        (
            "runtime",
            result["steady_state_event_pool_driver_creates"] == 0,
            "event pool driver creates in steady state",
        ),
        (
            "runtime",
            result["event_pool_growth_rejections"] == 0,
            "event pool growth rejections",
        ),
        ("runtime", result["event_pool_sealed"], "event pool not sealed"),
    )


def record_verdict(
    result: dict[str, Any],
    request: PlannedRequest,
    run: PlannedRun,
    comparison: Comparison,
) -> list[dict[str, str]]:
    """Run every check, record what failed in the artifact, and return it."""

    checks: tuple[_Check, ...] = (
        *_reference_checks(result, comparison),
        *_replay_checks(result, comparison),
        ("pressure", result["transfer_pressure_gate_passed"], "transfer pressure gate"),
        *_budget_checks(result, request, run),
        *_runtime_checks(result),
        (
            "store",
            result["store_modes_honoured"],
            "a store whose mode disables reads still served a hit",
        ),
    )
    failures = [
        {"category": category, "detail": detail}
        for category, satisfied, detail in checks
        if not satisfied
    ]
    result["failures"] = failures
    result["failure_categories"] = sorted({failure["category"] for failure in failures})
    result["passed"] = not failures
    return failures


def qualification_failed(
    request: PlannedRequest, failures: list[dict[str, str]]
) -> AssertionError:
    """The error a failing case raises, naming every category it failed in."""

    categories = sorted({failure["category"] for failure in failures})
    return AssertionError(
        f"{request.case.model_implementation} {request.case.family} numerical "
        f"qualification failed ({', '.join(categories)}): "
        + "; ".join(
            f"{failure['category']}: {failure['detail']}" for failure in failures
        )
        + f", artifact={request.result_path}"
    )
