"""Geometry enumeration and winner selection for the step sweep."""

from __future__ import annotations

import pytest
from torch import OutOfMemoryError

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
            StepSearchGeometryBuild(12, 1, "d0", 2.0, {"capture_lowering": 1.5}),
            StepSearchGeometryBuild(6, 2, "d1", 3.0),
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
            StepSearchGeometryBuild(12, 1, "d0", 2.0, {"capture_lowering": 1.5}),
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

    assert report.geometries == ()
    assert [point.status for point in report.points] == ["infeasible"] * 4
    assert all("out of memory" in (point.error or "") for point in report.points)
    assert report.winner(12 << 30, 1 << 30) is None
    assert sum("exhausted the device" in line for line in lines) == 2
    assert [line for line in lines if line.startswith("point 1/4")] == [
        "point 1/4: 2 x 1 @ 12 GiB -> infeasible"
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
