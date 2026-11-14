"""Installed CPU metadata resolvers for immutable pretrained model references."""

from collections.abc import Callable
from dataclasses import dataclass

from shadowspill_client import Model

from .contracts import versioned_id


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
            key not in getattr(resolved, field)
            or getattr(resolved, field)[key] != value
            for field in ("config", "precision")
            for key, value in getattr(requested, field).items()
        ):
            raise ValueError("reference resolver changed explicit model settings")
        if (
            requested.definition is not None
            and resolved.definition != requested.definition
        ):
            raise ValueError("reference resolver changed the explicit model definition")
        return resolved
