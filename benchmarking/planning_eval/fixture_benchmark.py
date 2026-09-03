"""Replay canonical PressureFit fixtures and measure implementation wall time."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shadowspill.ir import Program, ResidencySpec
from shadowspill.planner import (
    AdmissionFacts,
    PressureFitOptions,
    PressureFitResult,
    pressurefit,
)
from shadowspill.schema import artifact_schema
from shadowspill.simulator import DeviceSimulationConfig, SimulationConfig


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _fixture_paths(value: Path) -> tuple[Path, ...]:
    resolved = value.expanduser().resolve()
    if resolved.is_file():
        return (resolved,)
    if not resolved.is_dir():
        raise FileNotFoundError(f"fixture path does not exist: {resolved}")
    paths = tuple(sorted(resolved.glob("*.json")))
    if not paths:
        raise FileNotFoundError(f"fixture directory contains no JSON: {resolved}")
    return paths


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    program: Program
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    config: SimulationConfig
    options: PressureFitOptions
    admission: AdmissionFacts | None
    placement: AdmissionFacts | None


def _request(value: dict[str, Any]) -> ReplayRequest:
    request = value["request"]
    program = Program.from_dict(request["program"])
    initial = tuple(
        ResidencySpec.from_value(item, f"fixture.initial_residency[{index}]")
        for index, item in enumerate(request["initial_residency"])
    )
    final = tuple(
        ResidencySpec.from_value(item, f"fixture.final_residency[{index}]")
        for index, item in enumerate(request["final_residency"])
    )
    simulation = request["simulation_config"]
    config = SimulationConfig(
        devices=tuple(
            DeviceSimulationConfig(**device) for device in simulation["devices"]
        ),
        spill_capacity_bytes=simulation["spill_capacity_bytes"],
    )
    options = PressureFitOptions.from_dict(request["options"])
    admission_value = request.get("admission")
    admission = (
        None if admission_value is None else AdmissionFacts.from_dict(admission_value)
    )
    placement_value = request.get("placement")
    placement = (
        None if placement_value is None else AdmissionFacts.from_dict(placement_value)
    )
    return ReplayRequest(program, initial, final, config, options, admission, placement)


def _expected_value(result: PressureFitResult) -> dict[str, Any]:
    return {
        "schedule": result.schedule.to_dict(),
        "selections": [item.to_dict() for item in result.selections],
        "simulation": asdict(result.simulation),
        "diagnostics": result.diagnostics.stable_dict(),
    }


def _run_suite(paths: tuple[Path, ...], repeats: int) -> dict[str, Any]:
    fixtures = [json.loads(path.read_text()) for path in paths]
    for path, fixture in zip(paths, fixtures, strict=True):
        if fixture.get("schema") != artifact_schema("pressurefit_fixture"):
            raise ValueError(f"unsupported PressureFit fixture: {path}")
    requests = [_request(value) for value in fixtures]
    request_digests = [value["request_digest"] for value in fixtures]
    suite_digest = _digest(request_digests)
    samples: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for path, fixture, request in zip(paths, fixtures, requests, strict=True):
            result = pressurefit(
                request.program,
                initial_residency=request.initial_residency,
                final_residency=request.final_residency,
                config=request.config,
                options=request.options,
                admission=request.admission,
                placement=request.placement,
            )
            actual = _digest(_expected_value(result))
            if actual != fixture["expected_digest"]:
                raise AssertionError(
                    f"PressureFit output differs for {path}: "
                    f"expected={fixture['expected_digest']}, actual={actual}"
                )
        samples.append(time.perf_counter_ns() - started)
    median = round(statistics.median(samples))
    return {
        "suite_digest": suite_digest,
        "fixture_paths": [str(path) for path in paths],
        "request_digests": request_digests,
        "fixture_count": len(paths),
        "samples_ns": samples,
        "median_ns": median,
        "outputs_match": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay PressureFit golden inputs and require exact outputs."
    )
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="an earlier benchmark JSON used to calculate per-suite speedup",
    )
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        parser.error("--repeats must be positive")
    suites = [
        _run_suite(_fixture_paths(path), arguments.repeats)
        for path in arguments.fixtures
    ]
    baseline_by_digest: dict[str, int] = {}
    if arguments.baseline is not None:
        baseline_value = json.loads(
            arguments.baseline.expanduser().resolve().read_text()
        )
        if baseline_value.get("schema") != artifact_schema("pressurefit_benchmark"):
            parser.error("--baseline has an unsupported schema")
        baseline_by_digest = {
            value["suite_digest"]: int(value["median_ns"])
            for value in baseline_value["suites"]
        }
    for suite in suites:
        baseline_ns = baseline_by_digest.get(suite["suite_digest"])
        suite["baseline_median_ns"] = baseline_ns
        suite["speedup_over_baseline"] = (
            None if baseline_ns is None else baseline_ns / suite["median_ns"]
        )
    result = {
        "schema": artifact_schema("pressurefit_benchmark"),
        "implementation": "shadowspill.planner.pressurefit",
        "suites": suites,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
