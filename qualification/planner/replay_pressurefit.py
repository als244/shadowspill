"""Replay and time exact framework-free PressureFit fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shadowspill.ir import Program, ResidencySpec
from shadowspill.planner import PressureFitOptions, PressureFitResult, pressurefit
from shadowspill.planner.model import InitialPlacement
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.model import DeviceSimulationConfig


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _request(
    value: dict[str, Any],
) -> tuple[
    Program,
    tuple[ResidencySpec, ...],
    tuple[ResidencySpec, ...],
    SimulationConfig,
    PressureFitOptions,
]:
    request = value["request"]
    config = request["simulation_config"]
    options = request["options"]
    return (
        Program.from_dict(request["program"]),
        tuple(
            ResidencySpec.from_value(item, f"initial_residency[{index}]")
            for index, item in enumerate(request["initial_residency"])
        ),
        tuple(
            ResidencySpec.from_value(item, f"final_residency[{index}]")
            for index, item in enumerate(request["final_residency"])
        ),
        SimulationConfig(
            tuple(DeviceSimulationConfig(**item) for item in config["devices"]),
            config["host_capacity_bytes"],
        ),
        PressureFitOptions(
            initial_placement=InitialPlacement(options["initial_placement"]),
            residency_strategies=tuple(options["residency_strategies"]),
            prefetch_rules=tuple(options["prefetch_rules"]),
            evaluate_coalesced=options["evaluate_coalesced"],
            max_repair_attempts=options["max_repair_attempts"],
            workers=options["workers"],
        ),
    )


def _expected(result: PressureFitResult) -> dict[str, object]:
    return {
        "schedule": result.schedule.to_dict(),
        "selections": [item.to_dict() for item in result.selections],
        "simulation": asdict(result.simulation),
        "diagnostics": asdict(result.diagnostics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--python-authority",
        action="store_true",
        help="disable the compiled planner while retaining the compiled simulator",
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if arguments.python_authority:
        implementation = importlib.import_module("shadowspill.planner.pressurefit")
        setattr(implementation, "planner_library_path", lambda: None)  # noqa: B010

    for path in arguments.fixtures:
        fixture = json.loads(path.read_text())
        program, initial, final, config, options = _request(fixture)
        walls: list[float] = []
        result: PressureFitResult | None = None
        for _repeat in range(arguments.repeats):
            started = time.perf_counter()
            result = pressurefit(
                program,
                initial_residency=initial,
                final_residency=final,
                config=config,
                options=options,
            )
            walls.append(time.perf_counter() - started)
        assert result is not None
        actual_digest = _digest(_expected(result))
        print(
            _canonical(
                {
                    "fixture": str(path),
                    "mode": (
                        "python-authority"
                        if arguments.python_authority
                        else "compiled"
                    ),
                    "repeats": arguments.repeats,
                    "median_seconds": statistics.median(walls),
                    "minimum_seconds": min(walls),
                    "maximum_seconds": max(walls),
                    "expected_digest": fixture["expected_digest"],
                    "actual_digest": actual_digest,
                    "exact": actual_digest == fixture["expected_digest"],
                    "schedule_digest": result.schedule.digest,
                    "makespan_ns": result.simulation.makespan_ns,
                }
            )
        )


if __name__ == "__main__":
    main()
