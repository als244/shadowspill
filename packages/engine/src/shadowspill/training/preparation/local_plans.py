"""A local schedule proposal: numeric profiles and Plan metadata, never binaries."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from shadowspill.artifacts._values import freeze, plain
from shadowspill.artifacts.archive import Artifact
from shadowspill.artifacts.profiles import Profiles
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.training import ResolvedTraining

LOCAL_PLANS_MEDIA_TYPE = "application/vnd.shadowspill.local-plans+zip"


@dataclass(frozen=True)
class LocalPlans(Artifact):
    """Selected CPU schedules awaiting server coverage and physical admission.

    Unlike an executable TrainingBundle, this contains no native payloads. The
    receiving service supplies those from the authenticated preparation job.
    Rejected local searches are diagnostics, never new memory evidence.
    """

    artifact_kind: ClassVar[str] = "local_training_plans"
    metadata_limit_bytes: ClassVar[int | None] = 256 << 20
    payload_limit_bytes: ClassVar[int | None] = 0
    config: ResolvedTraining
    profiles: Mapping[str, Profiles]
    plans: Mapping[str, Mapping[str, Any]]
    candidates: Sequence[str]
    rejected: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.config, ResolvedTraining):
            raise TypeError("local Plans require a resolved configuration")
        if not self.config.preparation.reuse:
            raise ValueError("local Plans require the measured preparation identity")
        candidates = tuple(self.candidates)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("local Plans require distinct selected candidates")
        for key in (*self.profiles, *self.plans, *candidates, *self.rejected):
            if (
                not isinstance(key, str)
                or len(key) != 64
                or any(c not in "0123456789abcdef" for c in key)
            ):
                raise ValueError("local planning identities must be SHA-256 digests")
        object.__setattr__(self, "candidates", candidates)
        # Profiles intentionally remain editable outside this immutable proposal.
        object.__setattr__(
            self,
            "profiles",
            MappingProxyType(
                {
                    key: Profiles.from_dict(value.to_dict())
                    for key, value in self.profiles.items()
                }
            ),
        )
        object.__setattr__(self, "plans", freeze(self.plans))
        object.__setattr__(self, "rejected", freeze(self.rejected))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": self.config.to_dict(),
            "profiles": {key: value.to_dict() for key, value in self.profiles.items()},
            "plans": plain(self.plans),
            "candidates": list(self.candidates),
            "rejected": plain(self.rejected),
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        payloads: Mapping[str, bytes] | None = None,
    ) -> LocalPlans:
        if (
            payloads
            or set(value)
            != {
                "version",
                "config",
                "profiles",
                "plans",
                "candidates",
                "rejected",
            }
            or type(value["version"]) is not int
            or value["version"] != 1
        ):
            raise ValueError("unsupported local Plan proposal")
        return cls(
            ResolvedTraining.from_dict(value["config"]),
            {key: Profiles.from_dict(row) for key, row in value["profiles"].items()},
            value["plans"],
            value["candidates"],
            value["rejected"],
        )

    def write(self, path: str | Path) -> ArtifactRef:
        """Save explicitly and return the typed reference for ordinary upload."""
        self.save(path)
        with Path(path).open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        return ArtifactRef(
            "local-plans." + checksum,
            checksum,
            Path(path).stat().st_size,
            LOCAL_PLANS_MEDIA_TYPE,
        )
