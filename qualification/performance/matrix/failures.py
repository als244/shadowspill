"""Preserve evidence when a child exits before writing a verdict."""

from __future__ import annotations

import json
import re
import signal
from pathlib import Path

from shadowspill.schema import artifact_schema

# Cell logs may carry one leading "[<utc>] " stamp added by the matrix tee.
_PLAN_PROGRESS = re.compile(
    r"^(?:\[[^\]]*\] )?"
    r"(?P<phase>[A-Za-z0-9_]+): "
    r"(?P<state>failed|starting|completed)(?:\s+in\s+.*)?$"
)


def active_planning_phases(log_text: str) -> tuple[str, ...]:
    """Recover unfinished saved-phase workers from the current cell log."""

    active: list[str] = []
    for line in log_text.splitlines():
        match = _PLAN_PROGRESS.match(line)
        if match is None:
            continue
        phase = match.group("phase")
        if match.group("state") == "starting":
            active.append(phase)
            continue
        for index in range(len(active) - 1, -1, -1):
            if active[index] == phase:
                del active[index:]
                break
    return tuple(active)


def termination_signal(return_code: int) -> str | None:
    if return_code >= 0:
        return None
    try:
        return signal.Signals(-return_code).name
    except ValueError:
        return f"signal_{-return_code}"


def write_parent_failure(
    *,
    manifest_identity: str,
    return_code: int,
    log: Path,
    failure_path: Path,
) -> dict[str, object]:
    """Record failures a killed child had no opportunity to serialize."""

    text = log.read_text(errors="replace") if log.is_file() else ""
    active = active_planning_phases(text)
    signal_name = termination_signal(return_code)
    phase = active[-1] if active else None
    if signal_name is not None:
        message = f"qualification subprocess terminated by {signal_name}"
    else:
        message = f"qualification subprocess exited with status {return_code}"
    if phase is not None:
        message += f" during qualification phase {phase!r}"
    message += "; no Python traceback was available"
    progress_lines = [line for line in text.splitlines() if _PLAN_PROGRESS.match(line)]
    failure: dict[str, object] = {
        "schema": artifact_schema("full_model_subprocess_failure"),
        "identity": manifest_identity,
        "error_type": "SubprocessTermination",
        "error": message,
        "return_code": return_code,
        "termination_signal": signal_name,
        "active_planning_phases": active,
        "last_planning_progress": progress_lines[-1] if progress_lines else None,
        "log": str(log),
    }
    failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
    return failure
