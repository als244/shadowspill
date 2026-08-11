"""Golden PressureFit fixture serialization tests."""

from __future__ import annotations

import json
from pathlib import Path

from qualification.numerical.fixtures import (
    pressurefit_fixture,
    write_pressurefit_fixtures,
)
from shadowspill.planner import PressureFitOptions, PressureFitResult, pressurefit
from tests.planner._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
)
from verification.benchmark_pressurefit import _run_suite


def _result() -> PressureFitResult:
    program = exact_capacity_program()
    initial, final = exact_capacity_residency()
    result = pressurefit(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=PressureFitOptions(
            residency_strategies=("relaxed-stall",),
            prefetch_rules=("latest-safe",),
            evaluate_coalesced=False,
        ),
    )
    return result


def test_fixture_contains_complete_request_and_expected_result() -> None:
    result = _result()
    fixture = pressurefit_fixture(result, role="recurrent")

    assert fixture["request"]["program"] == result.program.to_dict()  # type: ignore[index]
    assert fixture["request"]["options"]["prefetch_rules"] == (  # type: ignore[index]
        "latest-safe",
    )
    assert fixture["expected"]["schedule"] == result.schedule.to_dict()  # type: ignore[index]
    assert fixture["expected"]["simulation"]["makespan_ns"] == 5_000  # type: ignore[index]


def test_fixture_file_is_byte_deterministic(tmp_path: Path) -> None:
    result = _result()
    first = write_pressurefit_fixtures(
        results=(result,),
        directory=tmp_path,
    )
    first_bytes = (tmp_path / "recurrent.json").read_bytes()
    second = write_pressurefit_fixtures(
        results=(result,),
        directory=tmp_path,
    )

    assert (tmp_path / "recurrent.json").read_bytes() == first_bytes
    assert first == second
    payload = json.loads(first_bytes)
    assert payload["request_digest"] == first[0]["request_digest"]
    assert payload["expected_digest"] == first[0]["expected_digest"]


def test_benchmark_replays_fixture_directly(tmp_path: Path) -> None:
    result = _result()
    write_pressurefit_fixtures(
        results=(result,),
        directory=tmp_path,
    )

    benchmark = _run_suite((tmp_path / "recurrent.json",), repeats=2)
    assert benchmark["outputs_match"] is True
    assert benchmark["fixture_count"] == 1
    assert isinstance(benchmark["suite_digest"], str)
    assert len(benchmark["samples_ns"]) == 2  # type: ignore[arg-type]
    assert benchmark["median_ns"] > 0  # type: ignore[operator]
