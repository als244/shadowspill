"""Installed phase-policy factories with CPU configuration and pinned sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from shadowspill_client._records import Value
from shadowspill_client.protocol.definitions import DefinitionSource

from .contracts import Configuration, versioned_id

type PolicyKind = Literal["partition", "chunking", "graph_pair_policy"]


@dataclass(frozen=True)
class PolicyDefinition:
    """Resolve configuration without creating a framework policy during preview."""

    id: str
    kind: PolicyKind
    config: Configuration
    create: Callable[[Any], Any]

    def __post_init__(self) -> None:
        versioned_id(self.id)
        if self.kind not in ("partition", "chunking", "graph_pair_policy"):
            raise ValueError("unknown phase-policy kind")
        if not callable(self.config) or not callable(self.create):
            raise TypeError("policy config and create must be callable")


@dataclass(frozen=True)
class ResolvedPolicy:
    definition: PolicyDefinition
    source: DefinitionSource
    config: Mapping[str, Value]

    def create(self) -> Any:
        from .registry import _configuration

        configured, settings = _configuration(self.definition.config, self.config)
        if settings != self.config:
            raise ValueError("policy configuration changed after resolution")
        return self.definition.create(configured)


@dataclass(frozen=True)
class PartitionConfig:
    mode: Literal["auto", "whole"] = "auto"

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "whole"):
            raise ValueError("partition mode must be auto or whole")


def _partition(config: PartitionConfig) -> str:
    return config.mode


def builtin_policy() -> tuple[PolicyDefinition, DefinitionSource]:
    return (
        PolicyDefinition(
            "shadowspill/partition@1", "partition", PartitionConfig, _partition
        ),
        DefinitionSource(
            "shadowspill",
            version("shadowspill"),
            sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
