"""Serialization for PressureFit options and selection diagnostics."""

from __future__ import annotations

from shadowspill.planner.diagnostics import PressureFitDiagnostics
from shadowspill.planner.request import InitialPlacement, PressureFitOptions
from shadowspill.planner.result import ResidentSlice

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
        "fetch_rules": list(options.fetch_rules),
        "evaluate_coalesced": options.evaluate_coalesced,
        "max_repair_attempts": options.max_repair_attempts,
        "workers": options.workers,
        "minimum_object_bytes_evict_eligible": (
            options.minimum_object_bytes_evict_eligible
        ),
    }


def _options_from_value(value: object, path: str) -> PressureFitOptions:
    data = _mapping(value, path)
    strategies = _list(data.get("residency_strategies"), f"{path}.residency_strategies")
    fetch = _list(data.get("fetch_rules"), f"{path}.fetch_rules")
    return PressureFitOptions(
        initial_placement=InitialPlacement(
            _string(data.get("initial_placement"), f"{path}.initial_placement")
        ),
        residency_strategies=tuple(
            _string(item, f"{path}.residency_strategies[{index}]")
            for index, item in enumerate(strategies)
        ),
        fetch_rules=tuple(
            _string(item, f"{path}.fetch_rules[{index}]")
            for index, item in enumerate(fetch)
        ),
        evaluate_coalesced=_boolean(
            data.get("evaluate_coalesced"), f"{path}.evaluate_coalesced"
        ),
        max_repair_attempts=_integer(
            data.get("max_repair_attempts"), f"{path}.max_repair_attempts"
        ),
        workers=_integer(data.get("workers"), f"{path}.workers"),
        minimum_object_bytes_evict_eligible=_integer(
            data.get("minimum_object_bytes_evict_eligible"),
            f"{path}.minimum_object_bytes_evict_eligible",
        ),
    )


def _resident_slice_from_value(value: object, path: str) -> ResidentSlice:
    data = _mapping(value, path)
    entries = _list(data.get("aliases"), f"{path}.aliases")
    return ResidentSlice(
        bytes=_integer(data.get("bytes"), f"{path}.bytes"),
        aliases=tuple(
            _string(item, f"{path}.aliases[{index}]")
            for index, item in enumerate(entries)
        ),
    )


def _pressurefit_diagnostics_from_value(
    value: object,
    path: str,
) -> PressureFitDiagnostics:
    return PressureFitDiagnostics.from_value(value, path)
