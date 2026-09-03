"""Strict configuration for repeatable PressureFit frontier sweeps."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from shadowspill.schema import artifact_schema

_SCHEMA = artifact_schema("pressurefit_frontier_config")
_SIZE_PATTERN = re.compile(r"^([1-9][0-9]*)(B|KiB|MiB|GiB|TiB)$")
_SIZE_MULTIPLIERS = {
    "B": 1,
    "KiB": 1 << 10,
    "MiB": 1 << 20,
    "GiB": 1 << 30,
    "TiB": 1 << 40,
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class BandwidthScale:
    """One exact rational multiplier applied to a Program's calibration."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator <= 0 or self.denominator <= 0:
            raise ValueError("bandwidth scale terms must be positive")
        if _gcd(self.numerator, self.denominator) != 1:
            raise ValueError("bandwidth scales must be reduced fractions")

    @property
    def label(self) -> str:
        return f"{self.numerator}over{self.denominator}x"

    def to_dict(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True, slots=True)
class FrontierGrid:
    """One Cartesian product of budgets and bandwidth scales."""

    name: str
    execution_budgets: tuple[int, ...]
    spill_budgets: tuple[int, ...]
    bandwidth_scales: tuple[BandwidthScale, ...]

    def __post_init__(self) -> None:
        if not self.name or _SAFE_NAME.fullmatch(self.name) is None:
            raise ValueError("grid name contains unsupported characters")
        for field, values in (
            ("execution_budgets", self.execution_budgets),
            ("spill_budgets", self.spill_budgets),
            ("bandwidth_scales", self.bandwidth_scales),
        ):
            if not values:
                raise ValueError(f"grid.{field} must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"grid.{field} contains duplicates")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "execution_budgets": [
                _format_size(item) for item in self.execution_budgets
            ],
            "spill_budgets": [_format_size(item) for item in self.spill_budgets],
            "bandwidth_scales": [item.to_dict() for item in self.bandwidth_scales],
        }


@dataclass(frozen=True, slots=True)
class TransferBandwidthBaseline:
    """One global concurrent transfer pair shared by every Program."""

    fetch_bytes_per_second: int
    evict_bytes_per_second: int
    provenance: str

    def __post_init__(self) -> None:
        if self.fetch_bytes_per_second <= 0 or self.evict_bytes_per_second <= 0:
            raise ValueError("base transfer bandwidths must be positive")
        if not self.provenance.strip():
            raise ValueError("transfer bandwidth provenance must be non-empty")

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "fetch_bytes_per_second": self.fetch_bytes_per_second,
            "evict_bytes_per_second": self.evict_bytes_per_second,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class FrontierConfig:
    """Complete immutable request for one planner-frontier baseline."""

    name: str
    expected_programs: int
    expected_points_per_program: int
    program_role: Literal["recurrent", "initial", "forward"]
    point_timeout_seconds: int
    max_point_attempts: int
    max_worker_restarts_per_program: int
    pressurefit_cache_mode: Literal["cold", "warm"]
    transfer_bandwidths: TransferBandwidthBaseline
    grids: tuple[FrontierGrid, ...]
    #: How much capacity a plan gives back at a time when its layout does not
    #: fit. Part of the config so two runs that differ only here are told
    #: apart by their config digest. Absent means the planner's own default.
    capacity_refinement_bytes: int | None = None
    #: How many repairs one candidate may spend. Absent means the planner's
    #: own default.
    max_repair_attempts: int | None = None

    def __post_init__(self) -> None:
        if not self.name or _SAFE_NAME.fullmatch(self.name) is None:
            raise ValueError("frontier name contains unsupported characters")
        for field, value in (
            ("expected_programs", self.expected_programs),
            ("expected_points_per_program", self.expected_points_per_program),
            ("point_timeout_seconds", self.point_timeout_seconds),
            ("max_point_attempts", self.max_point_attempts),
            (
                "max_worker_restarts_per_program",
                self.max_worker_restarts_per_program,
            ),
        ):
            if value <= 0:
                raise ValueError(f"{field} must be positive")
        if not self.grids:
            raise ValueError("grids must not be empty")

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "name": self.name,
            "expected_programs": self.expected_programs,
            "expected_points_per_program": self.expected_points_per_program,
            "program_role": self.program_role,
            "point_timeout_seconds": self.point_timeout_seconds,
            "max_point_attempts": self.max_point_attempts,
            "max_worker_restarts_per_program": (self.max_worker_restarts_per_program),
            "pressurefit_cache_mode": self.pressurefit_cache_mode,
            "capacity_refinement_bytes": self.capacity_refinement_bytes,
            "max_repair_attempts": self.max_repair_attempts,
            "transfer_bandwidths": self.transfer_bandwidths.to_dict(),
            "grids": [grid.to_dict() for grid in self.grids],
        }


def load_frontier_config(path: Path) -> FrontierConfig:
    """Load and strictly validate one versioned frontier configuration."""

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read frontier configuration {path}") from error
    data = _object(value, "config")
    _keys(
        data,
        {
            "schema",
            "name",
            "expected_programs",
            "expected_points_per_program",
            "program_role",
            "point_timeout_seconds",
            "max_point_attempts",
            "max_worker_restarts_per_program",
            "pressurefit_cache_mode",
            "transfer_bandwidths",
            "grids",
        },
        "config",
        optional={"capacity_refinement_bytes", "max_repair_attempts"},
    )
    if data.get("schema") != _SCHEMA:
        raise ValueError(f"config.schema must be {_SCHEMA!r}")
    grids = tuple(
        _grid(item, f"config.grids[{index}]")
        for index, item in enumerate(_array(data.get("grids"), "config.grids"))
    )
    config = FrontierConfig(
        name=_string(data.get("name"), "config.name"),
        expected_programs=_integer(
            data.get("expected_programs"), "config.expected_programs"
        ),
        expected_points_per_program=_integer(
            data.get("expected_points_per_program"),
            "config.expected_points_per_program",
        ),
        program_role=cast(
            Literal["recurrent", "initial", "forward"],
            _literal(
                data.get("program_role"),
                {"recurrent", "initial", "forward"},
                "config.program_role",
            ),
        ),
        point_timeout_seconds=_integer(
            data.get("point_timeout_seconds"), "config.point_timeout_seconds"
        ),
        max_point_attempts=_integer(
            data.get("max_point_attempts"), "config.max_point_attempts"
        ),
        max_worker_restarts_per_program=_integer(
            data.get("max_worker_restarts_per_program"),
            "config.max_worker_restarts_per_program",
        ),
        pressurefit_cache_mode=cast(
            Literal["cold", "warm"],
            _literal(
                data.get("pressurefit_cache_mode"),
                {"cold", "warm"},
                "config.pressurefit_cache_mode",
            ),
        ),
        transfer_bandwidths=_transfer_bandwidths(
            data.get("transfer_bandwidths"), "config.transfer_bandwidths"
        ),
        grids=grids,
        capacity_refinement_bytes=(
            None
            if data.get("capacity_refinement_bytes") is None
            else _integer(
                data.get("capacity_refinement_bytes"),
                "config.capacity_refinement_bytes",
            )
        ),
        max_repair_attempts=(
            None
            if data.get("max_repair_attempts") is None
            else _integer(data.get("max_repair_attempts"), "config.max_repair_attempts")
        ),
    )
    from .matrix import expand_grid_axes

    observed = len(expand_grid_axes(config.grids))
    if observed != config.expected_points_per_program:
        raise ValueError(
            "expanded point count does not match expected_points_per_program: "
            f"expected {config.expected_points_per_program}, observed {observed}"
        )
    return config


def _grid(value: object, path: str) -> FrontierGrid:
    data = _object(value, path)
    _keys(
        data,
        {"name", "execution_budgets", "spill_budgets", "bandwidth_scales"},
        path,
    )
    return FrontierGrid(
        name=_string(data.get("name"), f"{path}.name"),
        execution_budgets=tuple(
            _size(item, f"{path}.execution_budgets[{index}]")
            for index, item in enumerate(
                _array(data.get("execution_budgets"), f"{path}.execution_budgets")
            )
        ),
        spill_budgets=tuple(
            _size(item, f"{path}.spill_budgets[{index}]")
            for index, item in enumerate(
                _array(data.get("spill_budgets"), f"{path}.spill_budgets")
            )
        ),
        bandwidth_scales=tuple(
            _scale(item, f"{path}.bandwidth_scales[{index}]")
            for index, item in enumerate(
                _array(data.get("bandwidth_scales"), f"{path}.bandwidth_scales")
            )
        ),
    )


def _scale(value: object, path: str) -> BandwidthScale:
    data = _object(value, path)
    _keys(data, {"numerator", "denominator"}, path)
    return BandwidthScale(
        _integer(data.get("numerator"), f"{path}.numerator"),
        _integer(data.get("denominator"), f"{path}.denominator"),
    )


def _transfer_bandwidths(
    value: object,
    path: str,
) -> TransferBandwidthBaseline:
    data = _object(value, path)
    _keys(
        data,
        {"fetch_bytes_per_second", "evict_bytes_per_second", "provenance"},
        path,
    )
    return TransferBandwidthBaseline(
        fetch_bytes_per_second=_integer(
            data.get("fetch_bytes_per_second"), f"{path}.fetch_bytes_per_second"
        ),
        evict_bytes_per_second=_integer(
            data.get("evict_bytes_per_second"), f"{path}.evict_bytes_per_second"
        ),
        provenance=_string(data.get("provenance"), f"{path}.provenance"),
    )


def _size(value: object, path: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected a size string")
    match = _SIZE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"{path}: invalid size {value!r}")
    return int(match.group(1)) * _SIZE_MULTIPLIERS[match.group(2)]


def _format_size(value: int) -> str:
    for suffix, multiplier in reversed(tuple(_SIZE_MULTIPLIERS.items())):
        if value % multiplier == 0:
            return f"{value // multiplier}{suffix}"
    raise AssertionError("byte multiplier table has no unit divisor")


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected an object")
    return value


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected a string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected an integer")
    return value


def _literal(value: object, choices: set[str], path: str) -> str:
    selected = _string(value, path)
    if selected not in choices:
        raise ValueError(f"{path}: expected one of {sorted(choices)}")
    return selected


def _keys(
    data: dict[str, object],
    expected: set[str],
    path: str,
    optional: set[str] | None = None,
) -> None:
    allowed = expected | (optional or set())
    unknown = set(data) - allowed
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"{path}: unknown keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{path}: missing keys: {', '.join(sorted(missing))}")


def _gcd(left: int, right: int) -> int:
    while right:
        left, right = right, left % right
    return left


__all__ = [
    "BandwidthScale",
    "FrontierConfig",
    "FrontierGrid",
    "TransferBandwidthBaseline",
    "load_frontier_config",
]
