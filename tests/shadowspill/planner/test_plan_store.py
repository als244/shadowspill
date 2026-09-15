from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from shadowspill.ir import ShadowSpillProgram, TaskProfile, TaskSpec
from shadowspill.planner import (
    GenericPlanningOptions,
    SearchOptions,
)
from shadowspill.planner.admission import AdmissionFacts, TaskAdmissionSpec
from shadowspill.planner.admission.refinement import resolve_fixed_layout_selection
from shadowspill.planner.plan_store import PlanStore
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import (
    PressureFitOptions,
)
from shadowspill.store import StorePolicy

from ._examples import (
    COMPUTE,
    DEVICE,
    config,
    exact_capacity_program,
    exact_capacity_residency,
)

FEW_CANDIDATES = SearchOptions(
    generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
    algorithm=PressureFit(
        PressureFitOptions(
            residency_strategies=("relaxed-stall",),
            fetch_rules=("latest-safe",),
            evaluate_coalesced=False,
        )
    ),
)


def test_plan_store_preserves_the_complete_selection(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    cache = PlanStore(tmp_path)
    first = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=FEW_CANDIDATES,
    )
    second = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=FEW_CANDIDATES,
    )
    # Every option is part of a planned program's identity, so a search
    # configured differently in any respect is a different question and
    # reads no cached answer.
    varied = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=replace(
            FEW_CANDIDATES, generic=replace(FEW_CANDIDATES.generic, deterministic=True)
        ),
    )
    # Worker count is not one of them. It decides how long an answer takes,
    # not which answer is right, so the same question asked with more
    # threads reads the answer back rather than paying for it again.
    threaded = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=replace(FEW_CANDIDATES, workers=8),
    )

    assert not first.from_store
    assert second.from_store
    assert not varied.from_store
    assert threaded.from_store
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
        search_options=FEW_CANDIDATES,
    )
    # spelling out the library's default asks the same question
    spelled = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=replace(
            FEW_CANDIDATES,
            algorithm=PressureFit(
                replace(
                    FEW_CANDIDATES.algorithm.options,
                    resolution_options=[Fraction(n, 4) for n in range(5)],
                )
            ),
        ),
    )
    halves = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=replace(
            FEW_CANDIDATES,
            algorithm=PressureFit(
                replace(
                    FEW_CANDIDATES.algorithm.options,
                    resolution_options=("0", "1/2", "1"),
                )
            ),
        ),
    )
    again = cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=replace(
            FEW_CANDIDATES,
            algorithm=PressureFit(
                replace(
                    FEW_CANDIDATES.algorithm.options,
                    resolution_options=(Fraction(1), "1/2", 0),
                )
            ),
        ),
    )

    assert not first.from_store
    assert spelled.from_store
    assert not halves.from_store
    assert again.from_store
    records = [
        json.loads(path.read_text()) for path in tmp_path.rglob("selection.json")
    ]
    stored = sorted(
        json.dumps(item["search_options"]["algorithm"]["options"]["resolution_options"])
        for item in records
    )
    assert stored == [
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
        search_options=FEW_CANDIDATES,
    )
    fresh = PlanStore(tmp_path, policy=StorePolicy(read_enabled=False)).resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=FEW_CANDIDATES,
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
        search_options=FEW_CANDIDATES,
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
            search_options=FEW_CANDIDATES,
        )


def test_pressurefit_cache_validates_persisted_call_boundary(tmp_path: Path) -> None:
    initial, final = exact_capacity_residency()
    cache = PlanStore(tmp_path)
    cache.resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=FEW_CANDIDATES,
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
            search_options=FEW_CANDIDATES,
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
        search_options=FEW_CANDIDATES,
    )
    # the same request with a plan in hand is the same key, so it reads back
    # the stored answer rather than searching
    handed = cache.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=FEW_CANDIDATES,
        incumbent=first.result,
    )
    assert not first.from_store
    assert handed.from_store
    assert handed.result.schedule == first.result.schedule
    key = _key(
        program,
        initial,
        final,
        config(),
        None,
        None,
        FEW_CANDIDATES,
    )
    assert json.loads(cache.path(key).read_text())["incumbent"] is None

    # a store that first sees the request with a plan in hand records which
    other = PlanStore(tmp_path / "b")
    searched = other.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=FEW_CANDIDATES,
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
        search_options=FEW_CANDIDATES,
    )
    assert replanned.from_store
    assert replanned.result.schedule == first.result.schedule


def test_a_store_holding_a_worse_plan_answers_with_the_better_plan_in_hand(
    tmp_path: Path,
) -> None:
    """A request handed a plan never answers worse than it, stored or not."""

    initial, final = exact_capacity_residency()
    program = exact_capacity_program()
    every = PlanStore(tmp_path / "every").resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=SearchOptions(
            generic=GenericPlanningOptions(
                minimum_object_bytes_evict_eligible=0, deterministic=True
            )
        ),
    )
    # one candidate that plans this program a fifth slower than the best
    poor = GenericPlanningOptions(
        minimum_object_bytes_evict_eligible=0, deterministic=True
    )
    poor_search = PressureFitOptions(
        residency_strategies=("headroom-stall",),
        fetch_rules=("demand",),
        evaluate_coalesced=False,
    )
    cache = PlanStore(tmp_path / "poor")
    unaided = cache.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=SearchOptions(generic=poor, algorithm=PressureFit(poor_search)),
    )
    assert unaided.result.simulation.makespan_ns > every.result.simulation.makespan_ns

    # the store holds the poor plan; handed the better one, the request
    # searches again and answers with it
    handed = cache.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=SearchOptions(generic=poor, algorithm=PressureFit(poor_search)),
        incumbent=every.result,
    )
    assert not handed.from_store
    assert handed.result.schedule == every.result.schedule
    assert handed.result.diagnostics.selected_candidate_id == "incumbent"
    # and the store now holds the better plan for everyone after
    again = cache.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=SearchOptions(generic=poor, algorithm=PressureFit(poor_search)),
    )
    assert again.from_store
    assert again.result.schedule == every.result.schedule
    # a plan no better than the one on record leaves the record alone
    same = cache.resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=SearchOptions(generic=poor, algorithm=PressureFit(poor_search)),
        incumbent=unaided.result,
    )
    assert same.from_store
    assert same.result.schedule == every.result.schedule


def test_the_worker_count_reaches_the_search_it_was_given_to(tmp_path: Path) -> None:
    """A dropped `workers` is invisible: the plan is still right, just not
    searched the way the caller asked, and the diagnostics record the default.
    """

    initial, final = exact_capacity_residency()
    seen: list[int] = []

    class Counting(PressureFit):
        name = "counting"

        def __call__(self, program, **named):  # type: ignore[no-untyped-def]
            seen.append(named["workers"])
            return super().__call__(program, **named)

    PlanStore(tmp_path).resolve(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=SearchOptions(
            generic=FEW_CANDIDATES.generic,
            algorithm=Counting(FEW_CANDIDATES.resolved_algorithm.options),
            workers=1,
        ),
    )
    assert seen == [1]


def _placeable_program() -> ShadowSpillProgram:
    """One task and no objects: a layout the admission builder places as is."""

    return ShadowSpillProgram(
        devices=(DEVICE,),
        alias_groups=(),
        objects=(),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(TaskSpec("task", COMPUTE, "profile"),),
    )


def _facts(program: ShadowSpillProgram, pool_bytes: int) -> AdmissionFacts:
    return AdmissionFacts(
        "cuda_0",
        pool_bytes,
        pool_bytes,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )


def test_a_stored_plan_is_read_back_without_simulating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hit trusts the record: the simulator is not run for it."""

    import shadowspill.simulator.indexing as indexing

    initial, final = exact_capacity_residency()
    cache = PlanStore(tmp_path)
    request = dict(
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=FEW_CANDIDATES,
    )
    first = cache.resolve(exact_capacity_program(), **request)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a hit must not simulate")

    monkeypatch.setattr(indexing, "_run_projection", refuse)
    second = cache.resolve(exact_capacity_program(), **request)
    assert second.from_store
    assert second.key
    assert second.result.simulation == first.result.simulation
    assert second.result.simulation.interval_arrays is None


def test_a_certificate_is_written_beside_the_plan_and_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Certifying once writes the layout beside the plan; a later read for the
    same facts is served with it, and nothing is placed or simulated again."""

    import shadowspill.planner.admission.refinement as refinement

    program = _placeable_program()
    cache = PlanStore(tmp_path)
    request = dict(
        initial_residency=(),
        final_residency=(),
        config=config(),
        search_options=FEW_CANDIDATES,
    )
    first = cache.resolve(program, **request)
    assert first.certificate is None
    facts = _facts(program, config().devices[0].capacity_bytes)
    certified = resolve_fixed_layout_selection(
        config(), facts, lambda _config: first, certify=cache.certify
    )

    read_back = cache.resolve(program, **request)
    assert read_back.from_store
    assert read_back.certificate is not None
    assert read_back.certificate.layout == certified.admission.layout
    assert read_back.certificate.simulation == certified.admission.simulation

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a certified plan must not be placed again")

    monkeypatch.setattr(refinement, "build_fixed_layout_admission", refuse)
    served = resolve_fixed_layout_selection(config(), facts, lambda _config: read_back)
    assert served.admission.layout == certified.admission.layout
    assert served.attempts[0].accepted

    # Other facts are another certificate: this one is not used for them.
    other = _facts(program, config().devices[0].capacity_bytes + 8)
    with pytest.raises(AssertionError, match="placed again"):
        resolve_fixed_layout_selection(config(), other, lambda _config: read_back)


def test_a_recorded_verdict_is_served_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal is recorded under the plan's key and raised again from the
    record; with a plan to beat in hand the question is searched again."""

    from shadowspill.errors import PlanInfeasibleError

    initial, final = exact_capacity_residency()
    program = exact_capacity_program()
    cache = PlanStore(tmp_path)
    impossible = dict(
        initial_residency=initial,
        final_residency=final,
        config=config(capacity=8),
        search_options=FEW_CANDIDATES,
    )
    with pytest.raises(PlanInfeasibleError) as first:
        cache.resolve(program, **impossible)

    feasible = PlanStore(tmp_path / "feasible").resolve(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        search_options=FEW_CANDIDATES,
    )

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a recorded verdict must not be searched again")

    monkeypatch.setattr(PressureFit, "__call__", refuse)
    with pytest.raises(PlanInfeasibleError) as again:
        cache.resolve(program, **impossible)
    assert str(again.value) == str(first.value)
    assert again.value.kind == first.value.kind

    with pytest.raises(AssertionError, match="searched again"):
        cache.resolve(program, incumbent=feasible.result, **impossible)


def test_a_refused_miss_names_the_request_it_could_not_answer(tmp_path: Path) -> None:
    """The key alone says nothing a reader can compare with the store's records."""

    initial, final = exact_capacity_residency()
    program = exact_capacity_program()
    requested = config()
    device = requested.devices[0]
    with pytest.raises(LookupError) as refused:
        PlanStore(tmp_path, policy=StorePolicy.for_mode("require")).resolve(
            program,
            initial_residency=initial,
            final_residency=final,
            config=requested,
            search_options=FEW_CANDIDATES,
        )
    message = str(refused.value)
    assert f"program {program.digest[:12]}" in message
    assert f"capacity {device.capacity_bytes} B" in message
    assert f"fetch {device.fetch_bandwidth_bytes_per_second} B/s" in message
    assert "'require'" in message


def test_admission_facts_keep_their_digest_and_a_changed_copy_gets_a_new_one() -> None:
    program = _placeable_program()
    facts = _facts(program, 4096)
    first = facts.digest
    assert facts.digest == first and facts._digest_cache == [first]
    changed = replace(facts, pool_capacity_bytes=8192)
    assert changed.digest != first
    assert replace(facts, pool_capacity_bytes=4096).digest == first
