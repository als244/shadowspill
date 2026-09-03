from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from shadowspill.planner import PressureFitOptions
from shadowspill.planner.plan_store import PlanStore

from ._examples import config, exact_capacity_program, exact_capacity_residency

FEW_CANDIDATES = PressureFitOptions(
    minimum_object_bytes_evict_eligible=0,
    residency_strategies=("relaxed-stall",),
    fetch_rules=("latest-safe",),
    evaluate_coalesced=False,
)


def test_plan_store_preserves_the_complete_selection(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    cache = PlanStore(tmp_path)
    first = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )
    second = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )
    # Every option is part of a planned program's identity, so a search
    # configured differently in any respect is a different question and
    # reads no cached answer.
    varied = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=replace(FEW_CANDIDATES, workers=8),
    )

    assert not first.from_store
    assert second.from_store
    assert not varied.from_store
    assert second.result.schedule == first.result.schedule
    assert second.result.selections == first.result.selections
    assert second.result.simulation == first.result.simulation
    assert second.result.diagnostics == first.result.diagnostics


def test_plan_store_ignores_only_fresh_work_timings(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    first = PlanStore(tmp_path).resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )
    fresh = PlanStore(tmp_path, read_enabled=False).resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )

    assert not first.from_store
    assert not fresh.from_store
    assert fresh.result.schedule == first.result.schedule
    assert fresh.result.diagnostics.work.simulation_calls == (
        first.result.diagnostics.work.simulation_calls
    )


def test_pressurefit_cache_rejects_corrupt_evidence(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    cache = PlanStore(tmp_path)
    cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )
    path = next(cache.root.rglob("*.json"))
    value = json.loads(path.read_text())
    value["diagnostics"]["selection"]["makespan_ns"] += 1
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="invalid diagnostics"):
        cache.resolve(
            exact_capacity_program(),
            initial_residency=initial,
            final_residency=final,
            config=config(),
            options=FEW_CANDIDATES,
        )


def test_pressurefit_cache_validates_persisted_call_boundary(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    cache = PlanStore(tmp_path)
    cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )
    path = next(cache.root.rglob("*.json"))
    value = json.loads(path.read_text())
    value["initial_residency"] = []
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="initial_residency"):
        cache.resolve(
            exact_capacity_program(),
            initial_residency=initial,
            final_residency=final,
            config=config(),
            options=FEW_CANDIDATES,
        )
