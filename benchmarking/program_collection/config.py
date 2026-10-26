"""Strict JSON loading for collection settings, without importing a provider."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from workloads.models import catalog

from .settings import _SCHEMA, CollectionConfig, GeometryAxes, ModelSpec, ProfilingSpec

_SIZE_PATTERN = re.compile(r"^([1-9][0-9]*)(B|KiB|MiB|GiB|TiB)$")
_SIZE_MULTIPLIERS = {
    "B": 1,
    "KiB": 1 << 10,
    "MiB": 1 << 20,
    "GiB": 1 << 30,
    "TiB": 1 << 40,
}


def load_collection_config(path: Path) -> CollectionConfig:
    """Load and strictly validate one JSON collection configuration."""

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read collection configuration {path}") from error
    data = _object(value, "config")
    _keys(
        data,
        {
            "schema",
            "name",
            "seed",
            "expected_programs",
            "case_timeout_seconds",
            "max_attempts",
            "geometry",
            "models",
            "profiling",
        },
        "config",
    )
    if data.get("schema") != _SCHEMA:
        raise ValueError(f"config.schema must be {_SCHEMA!r}")
    geometry = _geometry(data.get("geometry"), "config.geometry")
    models_raw = _array(data.get("models"), "config.models")
    models = tuple(
        _model(item, f"config.models[{index}]") for index, item in enumerate(models_raw)
    )
    profiling = _profiling(data.get("profiling"), "config.profiling")
    return CollectionConfig(
        name=_string(data.get("name"), "config.name"),
        seed=_integer(data.get("seed"), "config.seed"),
        expected_programs=_optional_integer(
            data.get("expected_programs"), "config.expected_programs"
        ),
        case_timeout_seconds=_integer(
            data.get("case_timeout_seconds"), "config.case_timeout_seconds"
        ),
        max_attempts=_integer(data.get("max_attempts"), "config.max_attempts"),
        geometry=geometry,
        models=models,
        profiling=profiling,
    )


def _geometry(value: object, path: str) -> GeometryAxes:
    data = _object(value, path)
    _keys(
        data,
        {
            "tokens_per_microbatch",
            "sequence_lengths",
            "accumulation_rounds",
            "accumulation_steps",
        },
        path,
        required={"tokens_per_microbatch", "sequence_lengths"},
    )
    present = tuple(
        key for key in ("accumulation_rounds", "accumulation_steps") if key in data
    )
    if len(present) != 1:
        raise ValueError(f"{path} must define exactly one accumulation_rounds field")
    rounds_key = present[0]
    return GeometryAxes(
        tokens_per_microbatch=_integer_tuple(
            data.get("tokens_per_microbatch"), f"{path}.tokens_per_microbatch"
        ),
        sequence_lengths=_integer_tuple(
            data.get("sequence_lengths"), f"{path}.sequence_lengths"
        ),
        accumulation_rounds=_integer_tuple(
            data.get(rounds_key), f"{path}.accumulation_rounds"
        ),
    )


def _model(value: object, path: str) -> ModelSpec:
    data = _object(value, path)
    _keys(
        data,
        {"name", "model", "implementation", "geometry", "head_scratch_bytes"},
        path,
        required={"name", "model", "implementation"},
    )
    model = _string(data.get("model"), f"{path}.model")
    implementation = _string(data.get("implementation"), f"{path}.implementation")
    if model not in catalog.names():
        raise ValueError(f"{path}.model is not a catalog model")
    if implementation not in catalog.IMPLEMENTATIONS:
        raise ValueError(f"{path}.implementation is unsupported")
    geometry_value = data.get("geometry")
    return ModelSpec(
        name=_string(data.get("name"), f"{path}.name"),
        model=model,
        implementation=implementation,
        geometry=(
            None
            if geometry_value is None
            else _geometry(geometry_value, f"{path}.geometry")
        ),
        head_scratch_bytes=_optional_size(
            data.get("head_scratch_bytes"), f"{path}.head_scratch_bytes"
        ),
    )


def _profiling(value: object, path: str) -> ProfilingSpec:
    data = _object(value, path)
    _keys(
        data,
        {"device", "warmup_seconds", "measure_seconds", "external_headroom_bytes"},
        path,
    )
    return ProfilingSpec(
        device=_string(data.get("device"), f"{path}.device"),
        warmup_seconds=_seconds(data.get("warmup_seconds"), f"{path}.warmup_seconds"),
        measure_seconds=_seconds(
            data.get("measure_seconds"), f"{path}.measure_seconds"
        ),
        external_headroom_bytes=_integer(
            data.get("external_headroom_bytes"), f"{path}.external_headroom_bytes"
        ),
    )


def _seconds(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be seconds")
    return float(value)


def _keys(
    value: dict[str, Any],
    allowed: set[str],
    path: str,
    *,
    required: set[str] | None = None,
) -> None:
    missing = (allowed if required is None else required) - value.keys()
    extra = value.keys() - allowed
    if missing:
        raise ValueError(f"{path} is missing keys: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{path} has unknown keys: {', '.join(sorted(extra))}")


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _optional_integer(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _integer_tuple(value: object, path: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    )


def _size(value: object, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be bytes or a binary-size string")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{path} must be positive")
        return value
    if isinstance(value, str):
        match = _SIZE_PATTERN.fullmatch(value)
        if match is not None:
            return int(match.group(1)) * _SIZE_MULTIPLIERS[match.group(2)]
    raise ValueError(f"{path} must be bytes or a size such as '30GiB'")


def _optional_size(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _size(value, path)
