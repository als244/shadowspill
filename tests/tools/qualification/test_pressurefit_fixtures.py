"""Golden PressureFit fixture serialization tests."""

from __future__ import annotations

import json
from pathlib import Path

from shadowspill.planner import (
    PressureFitOptions,
    PressureFitResult,
    pressurefit,
)
from shadowspill.schema import artifact_schema
from tests.shadowspill.planner._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
)
from tests.shadowspill.planner.test_admission import _causal_facts
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
            fetch_rules=("latest-safe",),
            evaluate_coalesced=False,
            minimum_object_bytes_evict_eligible=0,
        ),
    )
    return result


def test_fixture_contains_complete_request_and_expected_result() -> None:
    result = _result()
    fixture = pressurefit_fixture(result, role="recurrent")

    assert fixture["request"]["program"] == result.program.to_dict()  # type: ignore[index]
    assert fixture["request"]["options"]["fetch_rules"] == (  # type: ignore[index]
        "latest-safe",
    )
    assert fixture["expected"]["schedule"] == result.schedule.to_dict()  # type: ignore[index]
    assert fixture["expected"]["simulation"]["makespan_ns"] == 5_000  # type: ignore[index]
    assert fixture["request"]["admission"] is None  # type: ignore[index]


def test_fixture_carries_the_physical_pressurefit_call_boundary() -> None:
    program = causal_program()
    schedule = causal_schedule()
    simulation = causal_config()
    admission = _causal_facts()
    result = pressurefit(
        program,
        initial_residency=schedule.initial_residency,
        final_residency=schedule.final_residency,
        config=simulation,
        options=PressureFitOptions(
            residency_strategies=("tight-stall",),
            fetch_rules=("latest-safe",),
            evaluate_coalesced=False,
            workers=1,
            minimum_object_bytes_evict_eligible=0,
        ),
        admission=admission,
    )

    fixture = pressurefit_fixture(result, role="recurrent")

    assert fixture["schema"] == artifact_schema("pressurefit_fixture")
    assert fixture["request"]["admission"] == admission.to_dict()  # type: ignore[index]
    assert fixture["request"]["placement"] is None  # type: ignore[index]


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
