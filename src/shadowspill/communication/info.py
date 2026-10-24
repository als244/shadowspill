"""Portable communication assumptions, separate from task profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from shadowspill.artifacts._values import freeze, plain, seconds
from shadowspill.artifacts.archive import Artifact


@dataclass(frozen=True)
class CommunicationInfo(Artifact):
    """Host transfer rates used by the initial single-rank offline path.

    Concurrent rates describe fetch/evict overlap. If omitted they equal the
    solo rates (independent full-duplex lanes); this assumption is saved.
    Topology is a mapping from node names to ordered logical rank IDs.
    """

    artifact_kind: ClassVar[str] = "communication_info"
    fetch_bytes_per_second: float
    evict_bytes_per_second: float
    fetch_concurrent_bytes_per_second: float | None = None
    evict_concurrent_bytes_per_second: float | None = None
    fetch_latency_seconds: float = 0.0
    evict_latency_seconds: float = 0.0
    topology: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for direction in ("fetch", "evict"):
            for suffix in ("bytes_per_second", "concurrent_bytes_per_second"):
                key = f"{direction}_{suffix}"
                rate = getattr(self, key)
                if rate is not None:
                    seconds(rate, key)
                    if rate < 1:
                        raise ValueError(f"{key} must be at least one byte/second")
            seconds(getattr(self, f"{direction}_latency_seconds"), "latency_seconds")
        ranks = [rank for group in self.topology.values() for rank in group]
        if len(ranks) != len(set(ranks)) or any(
            type(r) is not int or r < 0 for r in ranks
        ):
            raise ValueError("topology must assign each logical rank to one node")
        object.__setattr__(self, "topology", freeze(self.topology))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            name: plain(getattr(self, name))
            for name in self.__dataclass_fields__
            if name != "artifact_kind"
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> CommunicationInfo:
        if payloads:
            raise ValueError("CommunicationInfo cannot contain executable payloads")
        return cls(**value)
