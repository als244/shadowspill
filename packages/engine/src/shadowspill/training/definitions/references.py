"""Installed CPU metadata resolvers for immutable pretrained model references."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from shadowspill_client import Model
from shadowspill_client._records import Value

from .contracts import versioned_id


def _preserves(explicit: Value, resolved: Value) -> bool:
    """Mappings may gain defaults; supplied leaves and sequences remain fixed."""
    if isinstance(explicit, Mapping):
        return isinstance(resolved, Mapping) and all(
            key in resolved and _preserves(value, resolved[key])
            for key, value in explicit.items()
        )
    return explicit == resolved


@dataclass(frozen=True)
class ModelReferenceDefinition:
    """Resolve metadata only; downloading model weights belongs to initialization.

    ``accepts`` is a local predicate. The selected resolver may retrieve CPU
    metadata and must pin an immutable revision and an installed model ID.
    It preserves explicit overrides or rejects them with a concrete reason.
    """

    id: str
    accepts: Callable[[str], bool]
    resolve: Callable[[Model], Model]

    def __post_init__(self) -> None:
        versioned_id(self.id)
        if not callable(self.accepts) or not callable(self.resolve):
            raise TypeError("reference definition requires accepts/resolve functions")

    def pin(self, requested: Model) -> Model:
        weights = requested.initialize
        if weights is None or weights.kind != "pretrained":
            raise ValueError("metadata resolution requires pretrained weights")
        resolved = self.resolve(requested)
        if (
            not isinstance(resolved, Model)
            or resolved.definition is None
            or resolved.initialize is None
            or resolved.initialize.kind != "pretrained"
            or resolved.initialize.reference != weights.reference
            or resolved.initialize.revision is None
        ):
            raise ValueError(
                "reference resolver must pin model metadata and a revision"
            )
        if any(
            getattr(requested, name) is not None
            and getattr(requested, name) != getattr(resolved, name)
            for name in ("lora", "implementation", "trainable")
        ) or any(
            not _preserves(getattr(requested, field), getattr(resolved, field))
            for field in ("config", "precision")
        ):
            raise ValueError("reference resolver changed explicit model settings")
        if not _preserves(weights.metadata, resolved.initialize.metadata):
            raise ValueError("reference resolver changed explicit weight metadata")
        if (
            requested.definition is not None
            and resolved.definition != requested.definition
        ):
            raise ValueError("reference resolver changed the explicit model definition")
        return resolved
