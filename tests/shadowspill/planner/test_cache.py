from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadowspill.planner import PressureFitOptions
from shadowspill.planner.cache import PressureFitCache

from ._examples import config, exact_capacity_program, exact_capacity_residency

SMALL_PORTFOLIO = PressureFitOptions(
    residency_strategies=("relaxed-stall",),
    prefetch_rules=("latest-safe",),
    evaluate_coalesced=False,
)


def test_pressurefit_cache_preserves_the_complete_selection(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    cache = PressureFitCache(tmp_path)
    first = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=SMALL_PORTFOLIO,
    )
    second = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=PressureFitOptions(
            residency_strategies=SMALL_PORTFOLIO.residency_strategies,
            prefetch_rules=SMALL_PORTFOLIO.prefetch_rules,
            evaluate_coalesced=False,
            workers=8,
        ),
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert second.result.schedule == first.result.schedule
    assert second.result.selections == first.result.selections
    assert second.result.simulation == first.result.simulation
    assert second.result.diagnostics == first.result.diagnostics


def test_pressurefit_cache_ignores_only_fresh_work_timings(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    first = PressureFitCache(tmp_path).resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=SMALL_PORTFOLIO,
    )
    fresh = PressureFitCache(tmp_path, read_enabled=False).resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=SMALL_PORTFOLIO,
    )

    assert not first.cache_hit
    assert not fresh.cache_hit
    assert fresh.result.schedule == first.result.schedule
    assert fresh.result.diagnostics.work.simulation_calls == (
        first.result.diagnostics.work.simulation_calls
    )


def test_pressurefit_cache_rejects_corrupt_evidence(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    cache = PressureFitCache(tmp_path)
    cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=SMALL_PORTFOLIO,
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
            options=SMALL_PORTFOLIO,
        )


def test_pressurefit_cache_validates_persisted_call_boundary(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    cache = PressureFitCache(tmp_path)
    cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=SMALL_PORTFOLIO,
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
            options=SMALL_PORTFOLIO,
        )
