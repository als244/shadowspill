"""Resolve experiment variables and engine matrices."""

from __future__ import annotations

import json
import re
from pathlib import Path

from baselines.common.launch import engine_module, interpreter
from baselines.common.provenance import ROOT

SPEC_KEYS = {
    "schema_version",
    "output",
    "gpus",
    "python",
    "case_timeout",
    "settings",
    "sweeps",
    "summary_budgets_gib",
    "variables",
}


SWEEP_KEYS = {"name", "engine", "config", "python", "case_timeout"}


VARIABLE_NAME = re.compile(r"[a-z][a-z0-9_]*")


VARIABLE = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def substitute(value, values: dict[str, str]):
    """``value`` with every ``{name}`` its strings hold replaced by the
    variable's value; a reference to an undeclared variable is refused."""

    if isinstance(value, str):

        def replace(match):
            if match.group(1) not in values:
                raise ValueError(
                    f"{match.group(0)} is not a variable the spec declares"
                )
            return values[match.group(1)]

        return VARIABLE.sub(replace, value)
    if isinstance(value, list):
        return [substitute(item, values) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, values) for key, item in value.items()}
    return value


def variable_values(spec: dict) -> dict[str, str]:
    """The values a loaded spec's variables were given."""

    return {name: entry["value"] for name, entry in spec.get("variables", {}).items()}


def recorded_values(output: Path | None) -> dict[str, str]:
    """The variables' values an experiment's record holds, from its last run."""

    record = None if output is None else Path(output) / "experiment.json"
    if record is None or not record.is_file():
        return {}
    return variable_values(json.loads(record.read_text()).get("spec", {}))


def resolve_config(config: str, base: Path) -> str:
    """A matrix's path, from the spec's directory or else the repository's
    root, as the repository names it when it is inside."""

    path = Path(config)
    candidates = [path] if path.is_absolute() else [base / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            try:
                return str(resolved.relative_to(ROOT))
            except ValueError:
                return str(resolved)
    raise ValueError(f"no matrix at {config}")


def matrix_source(sweep: dict) -> Path:
    """The sweep's matrix file, from the repository root or as given."""

    path = ROOT / sweep["config"]
    return path if path.is_file() else Path(sweep["config"])


def sweep_matrix(path: Path, settings: dict, values: dict | None = None) -> dict:
    """The sweep's matrix with the experiment's settings in its defaults and
    the variables replaced; a matrix that names a setting itself is
    refused."""

    matrix = substitute(json.loads(Path(path).read_text()), values or {})
    cases = matrix.get("cases", [])
    if not isinstance(matrix.get("defaults", {}), dict) or not all(
        isinstance(case, dict) for case in cases
    ):
        raise ValueError(f"{path}: defaults and cases are mappings")
    grids = matrix.get("grid", {})
    named = set(matrix.get("defaults", {}))
    for axes in [*(grids if isinstance(grids, list) else [grids]), *cases]:
        named |= set(axes)
    shared = sorted(named & set(settings))
    if shared:
        raise ValueError(
            f"{path} names {', '.join(shared)}, which the experiment's settings set"
        )
    return {**matrix, "defaults": {**settings, **matrix.get("defaults", {})}}


def load_spec(
    path: Path, output: Path | None = None, variables: dict[str, str] | None = None
) -> dict:
    """The spec, checked: every variable has a value, from ``variables`` or
    the experiment's record, and every sweep names an engine and a matrix
    that leaves the experiment's settings to it, has a name of its own, and
    an interpreter resolved from its own entry, the spec's for its engine,
    or the environment the engine names."""

    spec = json.loads(Path(path).read_text())
    unknown = set(spec) - SPEC_KEYS
    if unknown or spec.get("schema_version") != 1:
        raise ValueError("expected schema_version=1 and gpus, python and sweeps")
    if output is not None:
        spec["output"] = str(Path(output).resolve())
    else:
        folder = Path(path).resolve().parent
        spec["output"] = str((folder / spec.get("output", "output")).resolve())
    if not isinstance(spec.get("gpus"), int) or spec["gpus"] < 1:
        raise ValueError("gpus must be a positive integer")
    pythons = spec.get("python", {})
    if not isinstance(pythons, dict):
        raise ValueError("python maps an engine to its interpreter")
    settings = spec.setdefault("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("settings maps a case's field to its value")
    declared = spec.get("variables", {})
    if not isinstance(declared, dict) or not all(
        isinstance(name, str)
        and VARIABLE_NAME.fullmatch(name)
        and isinstance(meaning, str)
        and meaning
        for name, meaning in declared.items()
    ):
        raise ValueError("variables maps each variable's name to what its value is")
    values = {**recorded_values(Path(spec["output"])), **(variables or {})}
    unknown = sorted(set(values) - set(declared))
    if variables and set(variables) - set(declared):
        raise ValueError(f"the spec declares no variable {unknown}")
    missing = sorted(set(declared) - set(values))
    if missing:
        raise ValueError(
            "give every variable the spec declares a value: "
            + "; ".join(f"--variable {name}=<{declared[name]}>" for name in missing)
        )
    values = {name: values[name] for name in declared}
    spec["variables"] = {
        name: {"description": declared[name], "value": values[name]}
        for name in declared
    }
    settings = spec["settings"] = substitute(settings, values)
    budgets = spec.get("summary_budgets_gib", [])
    if not isinstance(budgets, list) or any(
        isinstance(value, bool) or not isinstance(value, int | float) or value <= 0
        for value in budgets
    ):
        raise ValueError("summary_budgets_gib lists GPU-memory budgets in GiB")
    timeout = spec.get("case_timeout", 3600)
    if not isinstance(timeout, int | float) or timeout <= 0:
        raise ValueError("case_timeout must be positive")
    sweeps = spec.get("sweeps")
    if not isinstance(sweeps, list) or not sweeps:
        raise ValueError("at least one sweep is required")
    names = set()
    for sweep in sweeps:
        unknown = set(sweep) - SWEEP_KEYS
        if unknown or not sweep.get("engine") or not sweep.get("config"):
            raise ValueError("each sweep names an engine and a config")
        sweep.setdefault("name", f"{sweep['engine']}-{Path(sweep['config']).stem}")
        if sweep["name"] in names or "/" in sweep["name"]:
            raise ValueError(f"sweep names are distinct plain names: {sweep['name']}")
        names.add(sweep["name"])
        python = sweep.get("python", pythons.get(sweep["engine"]))
        if not python:
            python = engine_module(sweep["engine"]).ENVIRONMENT
        sweep["python"] = str(interpreter(python))
        sweep.setdefault("case_timeout", timeout)
        sweep["config"] = resolve_config(sweep["config"], Path(path).resolve().parent)
        sweep_matrix(matrix_source(sweep), settings, values)
    return spec


def spec_path_of(path: Path) -> str:
    """The spec's path, as the repository names it when it is inside."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)
