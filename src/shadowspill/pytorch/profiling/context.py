"""Value-sensitive input context for structural task profiling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from shadowspill.pytorch.capture.artifacts import GraphArtifact

_SCHEMA = "shadowspill.profile_input_context/v4"


@dataclass(frozen=True, slots=True)
class ProfileInputContext:
    """Authentic non-floating contents that may affect task measurements."""

    control_value_digests: tuple[str | None, ...]

    @classmethod
    def from_artifact(cls, artifact: GraphArtifact) -> ProfileInputContext:
        if len(artifact.input_provenance) != artifact.argument_count:
            raise ValueError("profile input context differs from task argument arity")
        return cls(
            tuple(
                _control_value_digest(item.representative_value)
                for item in artifact.input_provenance
            ),
        )

    @property
    def digest(self) -> str:
        payload = {
            "schema": _SCHEMA,
            "control_value_digests": self.control_value_digests,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

def profile_input_context_digest(artifact: object) -> str | None:
    """Return the occurrence context for graph tasks and none for opaque tasks."""

    return (
        ProfileInputContext.from_artifact(artifact).digest
        if isinstance(artifact, GraphArtifact)
        else None
    )


def _control_value_digest(value: object) -> str | None:
    if not isinstance(value, torch.Tensor):
        return None
    if value.is_floating_point() or value.is_complex():
        return None
    source = value.detach().to(device="cpu").contiguous()
    payload = source.numpy().tobytes(order="C")
    identity = {
        "dtype": str(source.dtype),
        "shape": tuple(source.shape),
        "bytes": hashlib.sha256(payload).hexdigest(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = ["ProfileInputContext", "profile_input_context_digest"]
