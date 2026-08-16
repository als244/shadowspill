"""Serialization for PressureFit options and selection diagnostics."""

from __future__ import annotations

from shadowspill.planner import (
    PressureFitDiagnostics,
    PressureFitOptions,
)
from shadowspill.planner.model import InitialPlacement

from .common import (
    _boolean,
    _integer,
    _list,
    _mapping,
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
    return PressureFitDiagnostics.from_value(value, path)
