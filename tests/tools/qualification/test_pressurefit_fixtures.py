"""Golden PressureFit fixture serialization tests."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarking.planning_eval.fixture_benchmark import _run_suite
from shadowspill.planner import (
    PressureFitOptions,
    PressureFitResult,
    pressurefit,
)
from tests.shadowspill.planner._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
)
from tests.shadowspill.planner.test_compiled_admission import _causal_topology
from tests.shadowspill.simulator.test_admission_accounting import (
    _config as causal_config,
)
from tests.shadowspill.simulator.test_admission_accounting import (
    _program as causal_program,
)
from tests.shadowspill.simulator.test_admission_accounting import (
    _schedule as causal_schedule,
)
from tools.qualification.pressurefit_fixtures import (
    pressurefit_fixture,
    write_pressurefit_fixtures,
)


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
    assert fixture["request"]["admission"] is None  # type: ignore[index]


def test_fixture_replays_the_physical_pressurefit_call_boundary(
    tmp_path: Path,
) -> None:
    program = causal_program()
    schedule = causal_schedule()
    simulation = causal_config()
    admission = _causal_topology()
    result = pressurefit(
        program,
        initial_residency=schedule.initial_residency,
        final_residency=schedule.final_residency,
        config=simulation,
        options=PressureFitOptions(
            residency_strategies=("tight-stall",),
            prefetch_rules=("latest-safe",),
            evaluate_coalesced=False,
            workers=1,
        ),
        admission=admission,
    )

    fixture = pressurefit_fixture(result, role="recurrent")

    assert fixture["schema"] == "shadowspill.pressurefit_fixture/v3"
    assert fixture["request"]["admission"] == admission.to_dict()  # type: ignore[index]
    write_pressurefit_fixtures(results=(result,), directory=tmp_path)
    replay = _run_suite((tmp_path / "recurrent.json",), repeats=1)
    assert replay["outputs_match"] is True


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
