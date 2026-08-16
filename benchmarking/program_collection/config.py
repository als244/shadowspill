"""Strict configuration schema for reusable Program collection."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_SCHEMA = "shadowspill.program_corpus_collection/v1"
_SIZE_PATTERN = re.compile(r"^([1-9][0-9]*)(B|KiB|MiB|GiB|TiB)$")
_SIZE_MULTIPLIERS = {
    "B": 1,
    "KiB": 1 << 10,
    "MiB": 1 << 20,
    "GiB": 1 << 30,
    "TiB": 1 << 40,
}


@dataclass(frozen=True, slots=True)
class GeometryAxes:
    """Ordered geometry axes expanded with divisibility validation."""

    tokens_per_microbatch: tuple[int, ...]
    sequence_lengths: tuple[int, ...]
    accumulation_rounds: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values in (
            ("tokens_per_microbatch", self.tokens_per_microbatch),
            ("sequence_lengths", self.sequence_lengths),
            ("accumulation_rounds", self.accumulation_rounds),
        ):
            if not values:
                raise ValueError(f"geometry.{name} must not be empty")
            if any(value <= 0 for value in values):
                raise ValueError(f"geometry.{name} values must be positive")
            if len(values) != len(set(values)):
                raise ValueError(f"geometry.{name} contains duplicates")

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens_per_microbatch": list(self.tokens_per_microbatch),
            "sequence_lengths": list(self.sequence_lengths),
            # Preserve the schema-v1 canonical digest of the collected dataset.
            "accumulation_steps": list(self.accumulation_rounds),
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One registered benchmark-model provider and optional axes."""

    name: str
    family: Literal["llama3", "qwen35", "olmoe"]
    implementation: Literal["pytorch", "mlops"]
    preset: Literal["throughput"] = "throughput"
    geometry: GeometryAxes | None = None
    head_scratch_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.name or re.search(r"[^A-Za-z0-9._-]", self.name):
            raise ValueError(
                "model name must contain only letters, digits, '.', '_', or '-'"
            )
        if self.family == "olmoe" and self.implementation == "pytorch":
            raise ValueError("pure-PyTorch OLMoE collection is currently unsupported")
        if self.head_scratch_bytes is not None and self.head_scratch_bytes <= 0:
            raise ValueError("model head_scratch_bytes must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "implementation": self.implementation,
            "preset": self.preset,
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "head_scratch_bytes": self.head_scratch_bytes,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """Runtime pool capacities and planning-visible budgets."""

    execution_pool_capacity_bytes: int
    spill_pool_capacity_bytes: int
    execution_budget_bytes: int
    spill_budget_bytes: int
    dynamic_scratch_reserve_bytes: int | None = None
    execution_device: int | None = None

    def __post_init__(self) -> None:
        if self.execution_pool_capacity_bytes <= 0:
            raise ValueError("execution pool capacity must be positive")
        if self.spill_pool_capacity_bytes <= 0:
            raise ValueError("spill pool capacity must be positive")
        if not 0 < self.execution_budget_bytes <= self.execution_pool_capacity_bytes:
            raise ValueError("execution budget exceeds its configured pool capacity")
        if not 0 < self.spill_budget_bytes <= self.spill_pool_capacity_bytes:
            raise ValueError("spill budget exceeds its configured pool capacity")
        if (
            self.dynamic_scratch_reserve_bytes is not None
            and self.dynamic_scratch_reserve_bytes < 0
        ):
            raise ValueError("dynamic scratch reserve must be non-negative")
        if self.execution_device is not None and self.execution_device < 0:
            raise ValueError("execution device must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_pool_capacity": self.execution_pool_capacity_bytes,
            "spill_pool_capacity": self.spill_pool_capacity_bytes,
            "execution_budget": self.execution_budget_bytes,
            "spill_budget": self.spill_budget_bytes,
            "dynamic_scratch_reserve": self.dynamic_scratch_reserve_bytes,
            "execution_device": self.execution_device,
        }


@dataclass(frozen=True, slots=True)
class PlanningSpec:
    """Planning behavior shared by all collected Programs."""

    optimizer_ordering: Literal["stage_interleaved", "tail"]
    allocation_probe_seeds: int
    allocation_probe_repetitions: int
    save_plan: bool
    force_fresh: bool
    overwrite_plan: bool
    implementation_revision: str | None

    def __post_init__(self) -> None:
        if self.allocation_probe_seeds <= 0:
            raise ValueError("allocation_probe_seeds must be positive")
        if self.allocation_probe_repetitions <= 0:
            raise ValueError("allocation_probe_repetitions must be positive")
        if self.overwrite_plan and not self.force_fresh:
            raise ValueError("overwrite_plan requires force_fresh")

    def to_dict(self) -> dict[str, object]:
        return {
            "optimizer_ordering": self.optimizer_ordering,
            "allocation_probe_seeds": self.allocation_probe_seeds,
            "allocation_probe_repetitions": self.allocation_probe_repetitions,
            "save_plan": self.save_plan,
            "force_fresh": self.force_fresh,
            "overwrite_plan": self.overwrite_plan,
            "implementation_revision": self.implementation_revision,
        }


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """Complete immutable request for one Program corpus collection."""

    name: str
    seed: int
    expected_programs: int | None
    case_timeout_seconds: int
    max_attempts: int
    geometry: GeometryAxes
    models: tuple[ModelSpec, ...]
    runtime: RuntimeSpec
    planning: PlanningSpec

    def __post_init__(self) -> None:
        if not self.name or re.search(r"[^A-Za-z0-9._-]", self.name):
            raise ValueError(
                "collection name must contain only letters, digits, '.', '_', or '-'"
            )
        if not self.models:
            raise ValueError("models must not be empty")
        names = tuple(model.name for model in self.models)
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        if self.expected_programs is not None and self.expected_programs <= 0:
            raise ValueError("expected_programs must be positive")
        if self.case_timeout_seconds <= 0:
            raise ValueError("case_timeout_seconds must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "name": self.name,
            "seed": self.seed,
            "expected_programs": self.expected_programs,
            "case_timeout_seconds": self.case_timeout_seconds,
            "max_attempts": self.max_attempts,
            "geometry": self.geometry.to_dict(),
            "models": [model.to_dict() for model in self.models],
            "runtime": self.runtime.to_dict(),
            "planning": self.planning.to_dict(),
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
            "runtime",
            "planning",
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
    runtime = _runtime(data.get("runtime"), "config.runtime")
    planning = _planning(data.get("planning"), "config.planning")
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
        runtime=runtime,
        planning=planning,
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
        key
        for key in ("accumulation_rounds", "accumulation_steps")
        if key in data
    )
    if len(present) != 1:
        raise ValueError(
            f"{path} must define exactly one accumulation_rounds field"
        )
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
        {
            "name",
            "family",
            "implementation",
            "preset",
            "geometry",
            "head_scratch_bytes",
        },
        path,
        required={"name", "family", "implementation"},
    )
    family = _string(data.get("family"), f"{path}.family")
    implementation = _string(data.get("implementation"), f"{path}.implementation")
    preset = _string(data.get("preset", "throughput"), f"{path}.preset")
    if family not in {"llama3", "qwen35", "olmoe"}:
        raise ValueError(f"{path}.family is unsupported")
    if implementation not in {"pytorch", "mlops"}:
        raise ValueError(f"{path}.implementation is unsupported")
    if preset != "throughput":
        raise ValueError(f"{path}.preset is unsupported")
    geometry_value = data.get("geometry")
    return ModelSpec(
        name=_string(data.get("name"), f"{path}.name"),
        family=family,  # type: ignore[arg-type]
        implementation=implementation,  # type: ignore[arg-type]
        preset=preset,  # type: ignore[arg-type]
        geometry=(
            None
            if geometry_value is None
            else _geometry(geometry_value, f"{path}.geometry")
        ),
        head_scratch_bytes=_optional_size(
            data.get("head_scratch_bytes"), f"{path}.head_scratch_bytes"
        ),
    )


def _runtime(value: object, path: str) -> RuntimeSpec:
    data = _object(value, path)
    _keys(
        data,
        {
            "execution_pool_capacity",
            "spill_pool_capacity",
            "execution_budget",
            "spill_budget",
            "dynamic_scratch_reserve",
            "execution_device",
        },
        path,
    )
    return RuntimeSpec(
        execution_pool_capacity_bytes=_size(
            data.get("execution_pool_capacity"), f"{path}.execution_pool_capacity"
        ),
        spill_pool_capacity_bytes=_size(
            data.get("spill_pool_capacity"), f"{path}.spill_pool_capacity"
        ),
        execution_budget_bytes=_size(
            data.get("execution_budget"), f"{path}.execution_budget"
        ),
        spill_budget_bytes=_size(data.get("spill_budget"), f"{path}.spill_budget"),
        dynamic_scratch_reserve_bytes=_optional_size(
            data.get("dynamic_scratch_reserve"),
            f"{path}.dynamic_scratch_reserve",
        ),
        execution_device=_optional_integer(
            data.get("execution_device"), f"{path}.execution_device"
        ),
    )


def _planning(value: object, path: str) -> PlanningSpec:
    data = _object(value, path)
    _keys(
        data,
        {
            "optimizer_ordering",
            "allocation_probe_seeds",
            "allocation_probe_repetitions",
            "save_plan",
            "force_fresh",
            "overwrite_plan",
            "implementation_revision",
        },
        path,
    )
    ordering = _string(data.get("optimizer_ordering"), f"{path}.optimizer_ordering")
    if ordering not in {"stage_interleaved", "tail"}:
        raise ValueError(f"{path}.optimizer_ordering is unsupported")
    return PlanningSpec(
        optimizer_ordering=ordering,  # type: ignore[arg-type]
        allocation_probe_seeds=_integer(
            data.get("allocation_probe_seeds"), f"{path}.allocation_probe_seeds"
        ),
        allocation_probe_repetitions=_integer(
            data.get("allocation_probe_repetitions"),
            f"{path}.allocation_probe_repetitions",
        ),
        save_plan=_boolean(data.get("save_plan"), f"{path}.save_plan"),
        force_fresh=_boolean(data.get("force_fresh"), f"{path}.force_fresh"),
        overwrite_plan=_boolean(data.get("overwrite_plan"), f"{path}.overwrite_plan"),
        implementation_revision=_optional_string(
            data.get("implementation_revision"),
            f"{path}.implementation_revision",
        ),
    )


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


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _optional_integer(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


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


__all__ = [
    "CollectionConfig",
    "GeometryAxes",
    "ModelSpec",
    "PlanningSpec",
    "RuntimeSpec",
    "load_collection_config",
]
