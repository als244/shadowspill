"""Immutable resolved phase settings; inspectable without the engine or models."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from .._records import Record, Value, integer, mapping, record, text
from ..config.preparation import BuildCase, PlanningOptions, ProfilingOptions
from .definitions import DefinitionSource

PREPARATION_MEDIA_TYPE = "application/vnd.shadowspill.preparation+zip"


@dataclass(frozen=True)
class ResolvedPreparation(Record):
    preset: str
    source: DefinitionSource
    cases: Sequence[BuildCase]
    profiling: ProfilingOptions
    planning: PlanningOptions
    origins: Mapping[str, Literal["explicit", "preset", "engine_default"]]
    reuse: Mapping[str, Value] = field(default_factory=dict)
    external_headroom_bytes: int = 512 << 20

    def __post_init__(self) -> None:
        text(self.preset, "preset")
        integer(self.external_headroom_bytes, "external_headroom_bytes")
        for name, kind in (
            ("source", DefinitionSource),
            ("profiling", ProfilingOptions),
            ("planning", PlanningOptions),
        ):
            object.__setattr__(self, name, record(kind, getattr(self, name), name))
        if not isinstance(self.cases, (list, tuple)) or not self.cases:
            raise ValueError("resolved preparation requires complete build cases")
        cases = tuple(record(BuildCase, item, "case") for item in self.cases)
        for index, case in enumerate(cases):
            if set(case.build.values) != {"partition", "chunking", "graph_pair_policy"}:
                raise ValueError("resolved build cases require every phase choice")
            if case in cases[:index]:
                raise ValueError("resolved build cases must be distinct")
        if (
            self.planning.microbatch_counts is None
            or self.planning.search is None
            or self.planning.workers is None
        ):
            raise ValueError("resolved planning requires counts and search settings")
        origins = dict(self.origins)
        if any(
            not isinstance(k, str)
            or not k
            or v not in ("explicit", "preset", "engine_default")
            for k, v in origins.items()
        ):
            raise ValueError("unknown preparation origin")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "origins", MappingProxyType(origins))
        object.__setattr__(self, "reuse", mapping(self.reuse, "reuse"))
