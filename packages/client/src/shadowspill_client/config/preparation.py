"""Portable phase controls; the engine remains the only builder and planner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from .._records import Record, Value, integer, mapping, number, plain, record, text


@dataclass(frozen=True)
class Policy(Record):
    """A server-installed policy, never an uploaded Python callable."""

    definition: str
    config: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text(self.definition, "policy definition")
        object.__setattr__(self, "config", mapping(self.config, "policy config"))


class _Omitted:
    pass


_OMITTED = _Omitted()


@dataclass(frozen=True, init=False)
class BuildOptions(Record):
    """Fixed policy choices. Omitted and explicitly unchunked are distinct."""

    values: Mapping[str, Value]

    def __init__(
        self,
        *,
        partition: Policy | Mapping[str, Any] | _Omitted | None = _OMITTED,
        chunking: Policy | Mapping[str, Any] | _Omitted | None = _OMITTED,
        graph_pair_policy: Policy | Mapping[str, Any] | _Omitted | None = _OMITTED,
    ) -> None:
        values = {}
        for name, value in (
            ("partition", partition),
            ("chunking", chunking),
            ("graph_pair_policy", graph_pair_policy),
        ):
            if value is not _OMITTED:
                values[name] = (
                    None if value is None else record(Policy, value, name).to_dict()
                )
        object.__setattr__(self, "values", mapping(values, "build options"))

    def to_dict(self) -> dict[str, Any]:
        return dict(plain(self.values))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("build options require an object")
        return cls(**value)


def positive_choices(value: Sequence[int] | None, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a nonempty sequence")
    choices = tuple(integer(v, name, minimum=1) for v in value)
    if len(set(choices)) != len(choices):
        raise ValueError(f"{name} must not contain duplicates")
    return choices


@dataclass(frozen=True)
class BuildCase(Record):
    """One physical input size and policies, with an optional per-buffer item cap."""

    microbatch_size: int
    build: BuildOptions = field(default_factory=BuildOptions)
    max_datums: int | None = None

    def __post_init__(self) -> None:
        integer(self.microbatch_size, "microbatch_size", minimum=1)
        if self.max_datums is not None:
            integer(self.max_datums, "max_datums", minimum=1)
        object.__setattr__(self, "build", record(BuildOptions, self.build, "build"))


@dataclass(frozen=True)
class BuildCandidates(Record):
    microbatch_sizes: Sequence[int] | None = None
    chunking: Sequence[Policy | None] | None = None
    graph_pair_policies: Sequence[Policy | None] | None = None
    cases: Sequence[BuildCase] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "microbatch_sizes",
            positive_choices(self.microbatch_sizes, "microbatch_sizes"),
        )
        for name in ("chunking", "graph_pair_policies", "cases"):
            values = getattr(self, name)
            if values is None:
                continue
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{name} must be a nonempty sequence")
            kind = BuildCase if name == "cases" else Policy
            choices = tuple(
                None if value is None and name != "cases" else record(kind, value, name)
                for value in values
            )
            if any(v in choices[:i] for i, v in enumerate(choices)):
                raise ValueError(f"duplicate {name} choices")
            object.__setattr__(self, name, choices)
        if self.cases is not None and any(
            value is not None
            for value in (
                self.microbatch_sizes,
                self.chunking,
                self.graph_pair_policies,
            )
        ):
            raise ValueError("complete build cases cannot be mixed with candidate axes")


@dataclass(frozen=True)
class ProfilingOptions(Record):
    """Wire projection of the engine's task ProfilingOptions, in seconds."""

    conditioning_seconds: float = 0.25
    conditioning_wall_seconds: float = 3.0
    conditioning_max_samples: int = 32
    minimum_samples: int = 3
    maximum_samples: int = 128
    measurement_seconds: float = 0.18
    measurement_wall_seconds: float = 2.0
    relative_mad_threshold: float = 0.03
    half_drift_threshold: float = 0.03

    def __post_init__(self) -> None:
        counts = {"conditioning_max_samples", "minimum_samples", "maximum_samples"}
        for name, value in self.to_dict().items():
            if name in counts:
                integer(value, name, minimum=1)
            else:
                number(value, name, minimum=0)
        if self.maximum_samples < self.minimum_samples:
            raise ValueError("maximum_samples must be >= minimum_samples")


@dataclass(frozen=True)
class PlanningOptions(Record):
    microbatch_counts: Sequence[int] | None = None
    search: Mapping[str, Value] | None = None
    workers: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "microbatch_counts",
            positive_choices(self.microbatch_counts, "microbatch_counts"),
        )
        if self.search is not None:
            # SearchOptions.to_dict() is the existing engine's portable question.
            # Its operational worker count is supplied separately here.
            object.__setattr__(self, "search", mapping(self.search, "search"))
        if self.workers is not None:
            integer(self.workers, "workers")


@dataclass(frozen=True)
class Preparation(Record):
    build: BuildOptions = field(default_factory=BuildOptions)
    candidates: BuildCandidates = field(default_factory=BuildCandidates)
    profiling: ProfilingOptions | None = None
    planning: PlanningOptions = field(default_factory=PlanningOptions)
    preset: str | None = None
    reuse: Mapping[str, Value] = field(default_factory=dict)
    external_headroom_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.external_headroom_bytes is not None:
            integer(self.external_headroom_bytes, "external_headroom_bytes")
        for name, kind in (
            ("build", BuildOptions),
            ("candidates", BuildCandidates),
            ("planning", PlanningOptions),
        ):
            object.__setattr__(self, name, record(kind, getattr(self, name), name))
        if self.profiling is not None:
            object.__setattr__(
                self, "profiling", record(ProfilingOptions, self.profiling, "profiling")
            )
        if self.preset is not None:
            text(self.preset, "preset")
        object.__setattr__(self, "reuse", mapping(self.reuse, "reuse"))
        fixed = self.build.values.keys()
        for axis, option in (
            ("chunking", "chunking"),
            ("graph_pair_policies", "graph_pair_policy"),
        ):
            if getattr(self.candidates, axis) is not None and option in fixed:
                raise ValueError(f"{option} is both a fixed value and a candidate axis")
        for case in self.candidates.cases or ():
            if overlap := fixed & case.build.values.keys():
                raise ValueError(f"build case repeats fixed choices: {sorted(overlap)}")
