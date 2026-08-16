"""Serialization for PressureFit options and selection diagnostics."""

from __future__ import annotations

from shadowspill.planner import (
    AdmissionRefinement,
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitOptions,
)
from shadowspill.planner.model import InitialPlacement

from .common import (
    _boolean,
    _integer,
    _list,
    _mapping,
    _optional_integer,
    _optional_string,
    _string,
)


def _options_to_dict(options: PressureFitOptions) -> dict[str, object]:
    return {
        "initial_placement": options.initial_placement.value,
        "residency_strategies": list(options.residency_strategies),
        "prefetch_rules": list(options.prefetch_rules),
        "evaluate_coalesced": options.evaluate_coalesced,
        "max_repair_attempts": options.max_repair_attempts,
        "workers": options.workers,
    }


def _options_from_value(value: object, path: str) -> PressureFitOptions:
    data = _mapping(value, path)
    strategies = _list(data.get("residency_strategies"), f"{path}.residency_strategies")
    prefetch = _list(data.get("prefetch_rules"), f"{path}.prefetch_rules")
    return PressureFitOptions(
        initial_placement=InitialPlacement(
            _string(data.get("initial_placement"), f"{path}.initial_placement")
        ),
        residency_strategies=tuple(
            _string(item, f"{path}.residency_strategies[{index}]")
            for index, item in enumerate(strategies)
        ),
        prefetch_rules=tuple(
            _string(item, f"{path}.prefetch_rules[{index}]")
            for index, item in enumerate(prefetch)
        ),
        evaluate_coalesced=_boolean(
            data.get("evaluate_coalesced"), f"{path}.evaluate_coalesced"
        ),
        max_repair_attempts=_integer(
            data.get("max_repair_attempts"), f"{path}.max_repair_attempts"
        ),
        workers=_integer(data.get("workers"), f"{path}.workers"),
    )


def _pressurefit_diagnostics_from_value(
    value: object,
    path: str,
) -> PressureFitDiagnostics:
    data = _mapping(value, path)
    candidates = _list(data.get("candidates"), f"{path}.candidates")
    refinements = _list(
        data.get("admission_refinements"), f"{path}.admission_refinements"
    )
    return PressureFitDiagnostics(
        selected_candidate_id=_string(
            data.get("selected_candidate_id"), f"{path}.selected_candidate_id"
        ),
        selected_selection_id=_string(
            data.get("selected_selection_id"), f"{path}.selected_selection_id"
        ),
        candidate_count=_integer(
            data.get("candidate_count"), f"{path}.candidate_count"
        ),
        valid_candidate_count=_integer(
            data.get("valid_candidate_count"), f"{path}.valid_candidate_count"
        ),
        selected_makespan_ns=_integer(
            data.get("selected_makespan_ns"), f"{path}.selected_makespan_ns"
        ),
        candidates=tuple(
            CandidateDiagnostic(
                candidate_id=_string(
                    item.get("candidate_id"), f"{path}.candidates[{index}].candidate_id"
                ),
                selection_id=_string(
                    item.get("selection_id"), f"{path}.candidates[{index}].selection_id"
                ),
                status=_string(
                    item.get("status"), f"{path}.candidates[{index}].status"
                ),
                makespan_ns=_optional_integer(
                    item.get("makespan_ns"), f"{path}.candidates[{index}].makespan_ns"
                ),
                schedule_digest=_optional_string(
                    item.get("schedule_digest"),
                    f"{path}.candidates[{index}].schedule_digest",
                ),
                failure_kind=_optional_string(
                    item.get("failure_kind"), f"{path}.candidates[{index}].failure_kind"
                ),
                failure_detail=_optional_string(
                    item.get("failure_detail"),
                    f"{path}.candidates[{index}].failure_detail",
                ),
                repair_attempts=_integer(
                    item.get("repair_attempts"),
                    f"{path}.candidates[{index}].repair_attempts",
                ),
            )
            for index, raw in enumerate(candidates)
            for item in (_mapping(raw, f"{path}.candidates[{index}]"),)
        ),
        admission_refinements=tuple(
            AdmissionRefinement(
                attempt=_integer(
                    item.get("attempt"),
                    f"{path}.admission_refinements[{index}].attempt",
                ),
                previous_object_capacity_bytes=_integer(
                    item.get("previous_object_capacity_bytes"),
                    f"{path}.admission_refinements[{index}].previous_object_capacity_bytes",
                ),
                required_additional_slack_bytes=_integer(
                    item.get("required_additional_slack_bytes"),
                    f"{path}.admission_refinements[{index}].required_additional_slack_bytes",
                ),
                reserve_increment_bytes=_integer(
                    item.get("reserve_increment_bytes"),
                    f"{path}.admission_refinements[{index}].reserve_increment_bytes",
                ),
                object_capacity_bytes=_integer(
                    item.get("object_capacity_bytes"),
                    f"{path}.admission_refinements[{index}].object_capacity_bytes",
                ),
            )
            for index, raw in enumerate(refinements)
            for item in (_mapping(raw, f"{path}.admission_refinements[{index}]"),)
        ),
        effective_object_capacity_bytes=_optional_integer(
            data.get("effective_object_capacity_bytes"),
            f"{path}.effective_object_capacity_bytes",
        ),
    )
