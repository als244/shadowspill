from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
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


def test_the_resolution_options_are_part_of_every_key(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    cache = PlanStore(tmp_path)
    first = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )
    # spelling out the library's default asks the same question
    spelled = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
        resolution_options=[Fraction(n, 4) for n in range(5)],
    )
    halves = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
        resolution_options=("0", "1/2", "1"),
    )
    again = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
        resolution_options=(Fraction(1), "1/2", 0),
    )

    assert not first.from_store
    assert spelled.from_store
    assert not halves.from_store
    assert again.from_store
    records = [
        json.loads(path.read_text()) for path in tmp_path.rglob("selection.json")
    ]
    assert sorted(json.dumps(item.get("resolution_options")) for item in records) == [
        '["0", "1/2", "1"]',
        '["0", "1/4", "1/2", "3/4", "1"]',
    ]


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


def test_the_plan_to_beat_is_provenance_not_identity(tmp_path: Path) -> None:
    """A request reads back the plan its search chose, whatever it was handed."""

    from shadowspill.planner.plan_store import _key

    initial, final = exact_capacity_residency()
    program = exact_capacity_program()
    cache = PlanStore(tmp_path / "a")
    first = cache.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )
    # the same request with a plan in hand is the same key, so it reads back
    # the stored answer rather than searching
    handed = cache.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
        incumbent=first.result,
    )
    assert not first.from_store
    assert handed.from_store
    assert handed.result.schedule == first.result.schedule
    shares = tuple(Fraction(n, 4) for n in range(5))
    key = _key(program, initial, final, config(), FEW_CANDIDATES, None, None, shares)
    assert json.loads(cache.path(key).read_text())["incumbent"] is None

    # a store that first sees the request with a plan in hand records which
    other = PlanStore(tmp_path / "b")
    searched = other.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
        incumbent=first.result,
    )
    assert not searched.from_store
    assert searched.result.diagnostics.selected_candidate_id == "incumbent"
    stored = json.loads(other.path(key).read_text())
    assert stored["incumbent"] == {
        "selections": [],
        "schedule_digest": first.result.schedule.digest,
    }
    # and a search of the same request without one reads that plan back
    replanned = other.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )
    assert replanned.from_store
    assert replanned.result.schedule == first.result.schedule
