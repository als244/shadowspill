"""Geometry enumeration and winner selection for the step sweep."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
from torch import OutOfMemoryError

from shadowspill.planner import StepDataOrdering
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.pytorch import StepSearchPoint, StepSearchReport, search_geometries
from shadowspill.schema import artifact_schema


def test_geometries_cover_every_divisor_largest_microbatch_first() -> None:
    admitted, skipped = search_geometries(12, sequence_length=1)
    assert admitted == ((12, 1), (6, 2), (4, 3), (3, 4), (2, 6), (1, 12))
    assert skipped == ()


def test_bounds_skip_with_reasons_rather_than_silently() -> None:
    admitted, skipped = search_geometries(
        12,
        sequence_length=1024,
        min_tokens_per_microbatch=3 * 1024,
        max_tokens_per_microbatch=6 * 1024,
    )
    assert admitted == ((6, 2), (4, 3), (3, 4))
    assert [(item[0], item[1]) for item in skipped] == [(12, 1), (2, 6), (1, 12)]
    assert all(item[2] for item in skipped)


def test_a_non_positive_total_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        search_geometries(0, sequence_length=1)


def _point(
    sequences: int, budget: int, status: str, makespan: float | None
) -> StepSearchPoint:
    return StepSearchPoint(
        sequences_per_microbatch=sequences,
        accumulation_count=12 // sequences,
        ordering=StepDataOrdering.depth_first(12 // sequences),
        execution_budget_bytes=budget,
        spill_budget_bytes=1,
        status=status,
        makespan_seconds=makespan,
        summary=None,
        error=None if status == "succeeded" else status,
        search_seconds=0.0,
    )


def test_report_totals_sum_the_work_where_it_was_paid() -> None:
    from shadowspill.pytorch import StepSearchGeometryBuild

    report = StepSearchReport(
        total_sequences_per_step=12,
        sequence_length=1024,
        budgets=((1, 1),),
        geometries=(
            StepSearchGeometryBuild(
                12,
                1,
                StepDataOrdering.depth_first(1),
                "d0",
                2.0,
                {"capture_lowering": 1.5},
            ),
            StepSearchGeometryBuild(6, 2, StepDataOrdering.depth_first(2), "d1", 3.0),
        ),
        points=(_point(12, 1, "succeeded", 4.0), _point(6, 1, "succeeded", 5.0)),
        skipped=(),
    )
    assert report.total_build_seconds == 5.0
    assert report.total_search_seconds == 0.0
    assert report.geometries[0].phase_seconds["capture_lowering"] == 1.5
    assert dict(report.geometries[1].phase_seconds) == {}


def test_the_report_serializes_for_post_hoc_analysis(tmp_path) -> None:
    import json

    from shadowspill.pytorch import StepSearchGeometryBuild

    report = StepSearchReport(
        total_sequences_per_step=12,
        sequence_length=1024,
        budgets=((1, 1),),
        geometries=(
            StepSearchGeometryBuild(
                12,
                1,
                StepDataOrdering.depth_first(1),
                "d0",
                2.0,
                {"capture_lowering": 1.5},
            ),
        ),
        points=(_point(12, 1, "succeeded", 4.0),),
        skipped=((3, 4, "below the minimum"),),
    )
    path = report.save(tmp_path / "search.json")
    payload = json.loads(path.read_text())
    assert payload["schema"] == artifact_schema("step_search_report")
    assert payload["geometries"][0]["phase_seconds"] == {"capture_lowering": 1.5}
    assert payload["points"][0]["status"] == "succeeded"
    assert payload["skipped"] == [[3, 4, "below the minimum"]]


def test_the_winner_is_the_fastest_succeeded_point_per_budget() -> None:
    report = StepSearchReport(
        total_sequences_per_step=12,
        sequence_length=1024,
        budgets=((1, 1), (2, 1)),
        geometries=(),
        points=(
            _point(12, 1, "infeasible", None),
            _point(6, 1, "succeeded", 4.0),
            _point(3, 1, "succeeded", 3.0),
            _point(6, 2, "search_exhausted", None),
            _point(3, 2, "infeasible", None),
        ),
        skipped=(),
    )
    winner = report.winner(1, 1)
    assert winner is not None and winner.sequences_per_microbatch == 3
    assert report.winner(2, 1) is None
    assert [item.execution_budget_bytes for item in report.winners] == [1]


def test_a_geometry_that_exhausts_the_device_marks_every_budget_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.errors import ProfilingError
    from shadowspill.pytorch import plan_step_search
    from shadowspill.pytorch import step_search as module

    def exhaust(*args: object, **kwargs: object) -> object:
        cause = OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")
        raise ProfilingError(
            f"ShadowSpill failed to profile structural contract abc123: {cause}"
        ) from cause

    monkeypatch.setattr(module, "make_step_program", exhaust)
    lines: list[str] = []
    report = plan_step_search(
        object(),  # type: ignore[arg-type]
        objective=None,
        opt=None,
        example_microbatches=lambda sequences, accumulation: (),
        total_sequences_per_step=2,
        sequence_length=1,
        budgets=[(12 << 30, 1 << 30), (16 << 30, 1 << 30)],
        runtime=None,  # type: ignore[arg-type]
        execution="execution",
        spill="spill",
        progress=lines.append,
    )

    # the 1 x 2 geometry has two orderings, and exhaustion is shared by both
    assert report.geometries == ()
    assert [point.status for point in report.points] == ["infeasible"] * 6
    assert [point.ordering.label for point in report.points] == [
        "1x1rp",
        "1x1rp",
        "2x1rp",
        "2x1rp",
        "1x2rp",
        "1x2rp",
    ]
    assert all("out of memory" in (point.error or "") for point in report.points)
    assert report.winner(12 << 30, 1 << 30) is None
    assert sum("exhausted the device" in line for line in lines) == 2
    assert [line for line in lines if line.startswith("point 1/6")] == [
        "point 1/6: 2 x 1 1x1rp @ 12 GiB -> infeasible"
    ]


def test_a_build_failure_that_is_not_exhaustion_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.errors import ProfilingError
    from shadowspill.pytorch import plan_step_search
    from shadowspill.pytorch import step_search as module

    def fail(*args: object, **kwargs: object) -> object:
        raise ProfilingError("an operator has no meta implementation")

    monkeypatch.setattr(module, "make_step_program", fail)
    with pytest.raises(ProfilingError, match="meta implementation"):
        plan_step_search(
            object(),  # type: ignore[arg-type]
            objective=None,
            opt=None,
            example_microbatches=lambda sequences, accumulation: (),
            total_sequences_per_step=1,
            sequence_length=1,
            budgets=[(12 << 30, 1 << 30)],
            runtime=None,  # type: ignore[arg-type]
            execution="execution",
            spill="spill",
        )


def test_a_point_the_planner_refuses_is_recorded_and_the_sweep_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.pytorch import plan_step_search
    from shadowspill.pytorch import step_search as module

    class Recurrent:
        transfer_bandwidths = TransferBandwidths(1_000, 2_000, provenance="stub")

    class Step:
        recurrent = Recurrent()
        digest = "d0"
        phase_timings_ns = (("total", 1),)

    def refuse(*args: object, **kwargs: object) -> object:
        raise RuntimeError("PressureFit problem rejected the selected facts")

    monkeypatch.setattr(module, "make_step_program", lambda *a, **k: Step())
    monkeypatch.setattr(module, "pressurefit_program", refuse)
    report = plan_step_search(
        object(),  # type: ignore[arg-type]
        objective=None,
        opt=None,
        example_microbatches=lambda sequences, accumulation: (),
        total_sequences_per_step=1,
        sequence_length=1,
        budgets=[(6 << 30, 1 << 30), (12 << 30, 1 << 30)],
        runtime=None,  # type: ignore[arg-type]
        execution="execution",
        spill="spill",
    )
    assert [point.status for point in report.points] == ["rejected"] * len(
        report.points
    )
    assert all("rejected the selected facts" in (p.error or "") for p in report.points)
    assert report.winner_plans == {}


def test_the_resolution_options_reach_every_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.planner import PressureFitInfeasibleError
    from shadowspill.pytorch import plan_step_search
    from shadowspill.pytorch import step_search as module

    class Recurrent:
        transfer_bandwidths = TransferBandwidths(1_000, 2_000, provenance="stub")

    class Step:
        recurrent = Recurrent()
        digest = "d0"
        phase_timings_ns = (("total", 1),)

    seen: list[object] = []

    def infeasible(*args: object, **kwargs: object) -> object:
        seen.append(kwargs["resolution_options"])
        raise PressureFitInfeasibleError("stub", kind="analytic_capacity")

    monkeypatch.setattr(module, "make_step_program", lambda *a, **k: Step())
    monkeypatch.setattr(module, "pressurefit_program", infeasible)
    report = plan_step_search(
        object(),  # type: ignore[arg-type]
        objective=None,
        opt=None,
        example_microbatches=lambda sequences, accumulation: (),
        total_sequences_per_step=1,
        sequence_length=1,
        budgets=[(12 << 30, 1 << 30)],
        runtime=None,  # type: ignore[arg-type]
        execution="execution",
        spill="spill",
        resolution_options=("1", "0"),
    )

    assert seen == [(Fraction(0), Fraction(1))]
    assert report.resolution_options == (Fraction(0), Fraction(1))
    assert report.to_dict()["resolution_options"] == ["0", "1"]
    assert [point.status for point in report.points] == ["infeasible"]
    # the calibration the build's program embeds is on the record, and no
    # override was given
    assert report.geometries[0].transfer_bandwidths == Recurrent.transfer_bandwidths
    serialized = report.to_dict()
    recorded = serialized["geometries"][0]["transfer_bandwidths"]
    assert recorded["fetch_bytes_per_second"] == 1_000
    assert report.transfer_bandwidths is None
    assert serialized["transfer_bandwidths"] is None


def test_a_pinned_calibration_reaches_every_point_and_the_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.planner import PressureFitInfeasibleError
    from shadowspill.pytorch import plan_step_search
    from shadowspill.pytorch import step_search as module

    class Recurrent:
        transfer_bandwidths = TransferBandwidths(1_000, 2_000, provenance="stub")

    class Step:
        recurrent = Recurrent()
        digest = "d0"
        phase_timings_ns = (("total", 1),)

    pinned = TransferBandwidths(26_000_000_000, 26_000_000_000, provenance="pin")
    seen: list[object] = []
    plan_stores: list[object] = []

    def infeasible(*args: object, **kwargs: object) -> object:
        seen.append(kwargs["transfer_bandwidths"])
        plan_stores.append(kwargs["plan_store_dir"])
        raise PressureFitInfeasibleError("stub", kind="analytic_capacity")

    monkeypatch.setattr(module, "make_step_program", lambda *a, **k: Step())
    monkeypatch.setattr(module, "pressurefit_program", infeasible)
    report = plan_step_search(
        object(),  # type: ignore[arg-type]
        objective=None,
        opt=None,
        example_microbatches=lambda sequences, accumulation: (),
        total_sequences_per_step=1,
        sequence_length=1,
        budgets=[(12 << 30, 1 << 30)],
        runtime=None,  # type: ignore[arg-type]
        execution="execution",
        spill="spill",
        transfer_bandwidths=pinned,
        plan_store_dir="plans-here",
    )

    assert seen == [pinned]
    assert plan_stores == ["plans-here"]
    assert report.transfer_bandwidths == pinned
    assert report.geometries[0].transfer_bandwidths == Recurrent.transfer_bandwidths
    assert report.to_dict()["transfer_bandwidths"]["provenance"] == "pin"


def test_resolution_options_that_are_not_valid_are_rejected_before_any_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shadowspill.pytorch import plan_step_search
    from shadowspill.pytorch import step_search as module

    def build(*args: object, **kwargs: object) -> object:
        raise AssertionError("no geometry may be built")

    monkeypatch.setattr(module, "make_step_program", build)
    with pytest.raises(ValueError, match="outside"):
        plan_step_search(
            object(),  # type: ignore[arg-type]
            objective=None,
            opt=None,
            example_microbatches=lambda sequences, accumulation: (),
            total_sequences_per_step=1,
            sequence_length=1,
            budgets=[(12 << 30, 1 << 30)],
            runtime=None,  # type: ignore[arg-type]
            execution="execution",
            spill="spill",
            resolution_options=("3/2",),
        )


def test_default_orderings_are_every_factor_pair_depth_first_first() -> None:
    from shadowspill.pytorch.step_search import default_orderings

    assert [item.label for item in default_orderings(8)] == [
        "8x1rp",
        "4x2rp",
        "2x4rp",
        "1x8rp",
    ]
    assert [item.label for item in default_orderings(1)] == ["1x1rp"]
    assert all(item.microbatches == 12 for item in default_orderings(12))


def test_the_winner_may_be_any_ordering_of_a_geometry() -> None:
    depth_first = _point(4, 10, "succeeded", 20.0)
    breadth = replace(depth_first, ordering=StepDataOrdering(1, 3))
    report = StepSearchReport(
        total_sequences_per_step=12,
        sequence_length=1,
        budgets=((10, 1),),
        geometries=(),
        points=(depth_first, replace(breadth, makespan_seconds=18.0)),
        skipped=(),
    )
    winner = report.winner(10, 1)
    assert winner is not None
    assert winner.ordering.label == "1x3rp"
    assert report.to_dict()["points"][1]["ordering_label"] == "1x3rp"


def test_each_budget_is_handed_the_best_plan_below_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budgets plan ascending, carry the best plan so far, and record an
    answer that was the handed-in plan by the budget it came from."""

    from shadowspill.pytorch import plan_step_search
    from shadowspill.pytorch import step_search as module

    class Recurrent:
        transfer_bandwidths = TransferBandwidths(1_000, 2_000, provenance="stub")

    class Step:
        recurrent = Recurrent()
        digest = "d0"
        phase_timings_ns = (("total", 1),)

    class Simulation:
        def __init__(self, makespan_ns: int) -> None:
            self.makespan_ns = makespan_ns

    class Diagnostics:
        def __init__(self, candidate: str) -> None:
            self.selected_candidate_id = candidate

    class Result:
        def __init__(self, makespan_ns: int, candidate: str) -> None:
            self.simulation = Simulation(makespan_ns)
            self.diagnostics = Diagnostics(candidate)

    class Plan:
        def __init__(self, makespan_ns: int, candidate: str) -> None:
            self.simulation = Simulation(makespan_ns)
            self.result = Result(makespan_ns, candidate)

    # what each budget's own search finds: 8 GiB is worse than 6 GiB, so it
    # answers with the 6 GiB plan; 10 GiB beats it
    found = {
        6 << 30: (100, "tight-stall/packed-fit"),
        8 << 30: (120, "x"),
        10 << 30: (90, "y"),
    }
    handed: list[tuple[int, object]] = []

    def search(*args: object, **kwargs: object) -> object:
        budget = kwargs["execution_budget"]
        assert isinstance(budget, int)
        incumbent = kwargs["incumbent"]
        handed.append((budget, incumbent))
        makespan, candidate = found[budget]
        if incumbent is not None and incumbent.simulation.makespan_ns <= makespan:
            return Plan(incumbent.simulation.makespan_ns, "incumbent")
        return Plan(makespan, candidate)

    monkeypatch.setattr(module, "make_step_program", lambda *a, **k: Step())
    monkeypatch.setattr(module, "summarize_selected_plan", lambda result: None)
    monkeypatch.setattr(module, "_graph_pair_outcomes", lambda result: ())
    monkeypatch.setattr(module, "pressurefit_program", search)
    report = plan_step_search(
        object(),  # type: ignore[arg-type]
        objective=None,
        opt=None,
        example_microbatches=lambda sequences, accumulation: (),
        total_sequences_per_step=1,
        sequence_length=1,
        budgets=[(10 << 30, 1 << 30), (6 << 30, 1 << 30), (8 << 30, 1 << 30)],
        runtime=None,  # type: ignore[arg-type]
        execution="execution",
        spill="spill",
    )

    # ascending, the first with nothing in hand, then the best so far
    assert [budget for budget, _ in handed] == [6 << 30, 8 << 30, 10 << 30]
    assert handed[0][1] is None
    assert handed[1][1] is not None and handed[1][1].simulation.makespan_ns == 100
    assert handed[2][1] is not None and handed[2][1].simulation.makespan_ns == 100
    points = {point.execution_budget_bytes: point for point in report.points}
    assert points[6 << 30].incumbent_budget_bytes is None
    assert points[8 << 30].incumbent_budget_bytes == 6 << 30
    assert points[8 << 30].makespan_seconds == 100 / 1e9
    assert points[10 << 30].incumbent_budget_bytes is None
    assert points[10 << 30].makespan_seconds == 90 / 1e9
    serialized = report.to_dict()["points"]
    assert isinstance(serialized, list)
    assert [item["incumbent_budget_bytes"] for item in serialized] == [
        None,
        6 << 30,
        None,
    ]

    # the winning plan of each budget is kept for the run that follows
    assert sorted(report.winner_plans) == [
        (6 << 30, 1 << 30),
        (8 << 30, 1 << 30),
        (10 << 30, 1 << 30),
    ]
    assert report.winner_plans[(8 << 30, 1 << 30)].simulation.makespan_ns == 100
    assert report.winner_plans[(10 << 30, 1 << 30)].simulation.makespan_ns == 90
    assert "winner_plans" not in report.to_dict()

    # every budget alone: nothing is handed in, and 8 GiB keeps its own answer
    handed.clear()
    alone = plan_step_search(
        object(),  # type: ignore[arg-type]
        objective=None,
        opt=None,
        example_microbatches=lambda sequences, accumulation: (),
        total_sequences_per_step=1,
        sequence_length=1,
        budgets=[(6 << 30, 1 << 30), (8 << 30, 1 << 30)],
        runtime=None,  # type: ignore[arg-type]
        execution="execution",
        spill="spill",
        incumbents=False,
    )
    assert [incumbent for _, incumbent in handed] == [None, None]
    assert [point.makespan_seconds for point in alone.points] == [100 / 1e9, 120 / 1e9]
    assert all(point.incumbent_budget_bytes is None for point in alone.points)
