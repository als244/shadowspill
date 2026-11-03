"""Independent saved-state cuts for the variants a checkpoint region offers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import torch

from ..capture.retention import RetentionPolicy


@dataclass(frozen=True, slots=True)
class GraphPairPolicy:
    """Choose each variant's operator retention policy, independently.

    ``save`` uses cost-based removals from backward-required forward state.
    ``recompute`` starts from checkpoint inputs and may retain selected results.
    These are capture decisions; neither policy chooses storage residency or
    which variant a stage executes. Planning chooses among measured variants.
    """

    save: RetentionPolicy = field(default_factory=RetentionPolicy)
    recompute: RetentionPolicy = field(
        default_factory=lambda: RetentionPolicy(default="recompute")
    )

    def __post_init__(self) -> None:
        for name in ("save", "recompute"):
            if not isinstance(getattr(self, name), RetentionPolicy):
                raise TypeError(f"{name} must be a RetentionPolicy")

    def variant(self, name: str) -> RetentionPolicy:
        if name == "save":
            return self.save
        if name == "recompute":
            return self.recompute
        raise ValueError(f"unknown graph-pair variant {name}; choose save or recompute")

    def identity(self) -> dict[str, object]:
        return {"save": self.save.identity(), "recompute": self.recompute.identity()}

    def operator_classes(
        self, graph_module: torch.fx.GraphModule
    ) -> dict[str, dict[str, str]]:
        return {
            "save": self.save.operator_classes(graph_module),
            "recompute": self.recompute.operator_classes(graph_module),
        }

    def digest(self, graph_module: torch.fx.GraphModule) -> str:
        encoded = json.dumps(
            {**self.identity(), "operators": self.operator_classes(graph_module)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = ["GraphPairPolicy"]
