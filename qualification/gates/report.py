"""Compact summaries from the suite and matrix result artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import RESULTS
from .run import GateOutcome

_NAMED_FAILURE_LIMIT = 15
_PYTEST_TALLY = re.compile(r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed)")


def _suite_report(log: Path) -> list[str]:
    """What the unit suite counted, from its own closing line."""

    try:
        text = log.read_text()
    except OSError:
        return ["    the suite wrote no readable log"]
    lines = text.splitlines()
    tallies: list[tuple[str, str]] = []
    for line in reversed(lines):
        tallies = _PYTEST_TALLY.findall(line)
        if tallies:
            break
    if not tallies:
        rows = ["    the suite reported no counts"]
    else:
        counts = {name: int(count) for count, name in tallies}
        counts.pop("deselected", None)
        rows = ["    " + ", ".join(f"{count} {name}" for name, count in counts.items())]
    named = [line.strip() for line in lines if line.startswith(("FAILED ", "ERROR "))]
    rows.extend(f"      {line}" for line in named[:_NAMED_FAILURE_LIMIT])
    if len(named) > _NAMED_FAILURE_LIMIT:
        rows.append(f"      ... and {len(named) - _NAMED_FAILURE_LIMIT} more, in {log}")
    return rows


def _numerical_report(directory: Path, *, network: bool = False) -> list[str]:
    """Which correctness cells agreed with their reference."""

    summary = directory / "summary.json"
    try:
        report = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return [f"    no readable summary at {summary}"]
    if report.get("skipped"):
        return [f"    skipped: {report.get('reason', 'no peer configured')}"]
    cases = report.get("cases", [])
    passed = sum(bool(case.get("passed")) for case in cases)
    suffix = " over the network" if network else ""
    rows = [f"    {passed}/{len(cases)} cells passed{suffix}"]
    rows.extend(
        f"      {case.get('implementation')}_{case.get('family')}: FAILED"
        for case in cases
        if not case.get("passed")
    )
    return rows


def _performance_report(directory: Path) -> list[str]:
    """Throughput and planning time per cell, measured against predicted."""

    summary = directory / "summary.json"
    try:
        report = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return [f"    no readable summary at {summary}"]
    if report.get("skipped"):
        return [f"    skipped: {report.get('reason', 'no peer configured')}"]
    cells = report.get("cells", [])
    results: list[tuple[str, dict[str, Any]]] = []
    for cell in cells:
        artifact = Path(str(cell.get("artifact", "")))
        try:
            results.append(
                (str(cell.get("identity")), json.loads(artifact.read_text()))
            )
        except (OSError, json.JSONDecodeError):
            continue
    if not results:
        return ["    no cell artifacts to report"]

    passed = sum(1 for _, value in results if value.get("passed"))
    rows = [f"    {passed}/{len(results)} cells passed", ""]
    rows.append(
        f"    {'cell':<14} {'real s':>8} {'real tok/s':>11} {'sim s':>8} "
        f"{'sim tok/s':>10} {'sim err':>8} {'plan s':>8}"
    )
    for identity, value in results:
        tokens = float(value["manifest"]["tokens_per_step"])
        simulated = float(value["predicted_makespan_seconds"])
        rows.append(
            f"    {identity:<14} {float(value['median_step_seconds']):8.3f} "
            f"{float(value['median_tokens_per_second']):11.1f} {simulated:8.3f} "
            f"{tokens / simulated:10.1f} "
            f"{float(value['simulator_relative_error']) * 100:+7.2f}% "
            f"{float(value['planning_seconds']):8.1f}"
        )

    phases: dict[str, dict[str, float]] = {}
    for identity, value in results:
        for phase, seconds in value.get("phase_seconds", {}).items():
            if phase != "total":
                phases.setdefault(phase, {})[identity] = float(seconds)
    ranked = sorted(
        phases.items(), key=lambda item: -max(item[1].values(), default=0.0)
    )
    material = [(name, times) for name, times in ranked if max(times.values()) >= 1.0]
    if material:
        rows.extend(["", "    planning time by phase, seconds"])
        names = [identity for identity, _ in results]
        width = max(len(phase) for phase, _ in material) + 2
        rows.append(
            "    " + f"{'phase':<{width}}" + "".join(f"{name:>14}" for name in names)
        )
        for phase, times in material:
            rows.append(
                f"    {phase:<{width}}"
                + "".join(f"{times.get(name, 0.0):>14.1f}" for name in names)
            )
    return rows


def _summary(line: str, outcomes: Sequence[GateOutcome], run: str) -> str:
    width = max(len(outcome.name) for outcome in outcomes)
    rows = [line]
    for outcome in outcomes:
        verdict = "PASS" if outcome.passed else f"FAIL ({outcome.returncode})"
        rows.append(
            f"  {outcome.name:<{width}}  {verdict:<10} "
            f"{outcome.seconds / 60:6.1f} min  {outcome.log}"
        )
        if outcome.name == "suite":
            rows.extend(_suite_report(outcome.log))
        elif outcome.name in ("numerical", "numerical_ssd"):
            rows.extend(_numerical_report(RESULTS / f"{outcome.name}_{run}"))
        elif outcome.name == "performance":
            rows.extend(_performance_report(RESULTS / f"performance_{run}"))
        elif outcome.name == "remote":
            rows.extend(_numerical_report(RESULTS / f"remote_{run}", network=True))
        elif outcome.name == "remote_perf":
            rows.extend(_performance_report(RESULTS / f"remote_perf_{run}"))
    return "\n".join(rows)
