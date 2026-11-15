"""Portable save settings, separate from preparation and mutable state."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from .._records import Record, Value, mapping, text


@dataclass(frozen=True)
class SaveSettings(Record):
    name: str | None = None
    metadata: Mapping[str, Value] = field(default_factory=dict)
    weights: Literal["master", "compute"] = "master"
    kind: Literal["checkpoint", "weights"] = "checkpoint"
    format: str = "shadowspill"

    def __post_init__(self) -> None:
        if self.kind not in ("checkpoint", "weights"):
            raise ValueError("unknown save kind")
        text(self.format, "save format")
        if len(self.format) > 128 or (
            self.kind == "checkpoint" and self.format != "shadowspill"
        ):
            raise ValueError("unsupported or oversized save format")
        if self.name is not None:
            text(self.name, "checkpoint name")
            if self.name != self.name.strip() or len(self.name) > 256:
                raise ValueError(
                    "checkpoint name must be trimmed and at most 256 characters"
                )
        if self.weights not in ("master", "compute"):
            raise ValueError("checkpoint weights must be master or compute")
        object.__setattr__(
            self, "metadata", mapping(self.metadata, "checkpoint metadata")
        )
