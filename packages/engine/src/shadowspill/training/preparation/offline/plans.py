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
    """Schedules planned locally, for the service to check and run.

    `PlanningInputs.plan` returns one, and `write` saves it for upload. It
    holds numeric profiles and Plan records but no executable code: the
    service supplies that from its own preparation, then checks every
    Program, schedule, input coverage and joint memory admission. Rejected
    searches are diagnostics, never memory evidence.

    Attributes
    ----------
    config : ResolvedTraining
        The settings planned for.
    profiles : mapping of str to Profiles
        The profiles planned with, by template ID.
    plans : mapping of str to mapping
        Each request's Plan, in `Plan.to_dict` form, by request ID.
    candidates : tuple of str
        The selected candidates, each with a Plan for every request.
    rejected : mapping of str to str
        Why planning failed, by request ID.
    """

    artifact_kind: ClassVar[str] = "local_training_plans"
    shared_metadata: ClassVar[bool] = True
    # The expanded family can be large even when shared tables are compact.
    # Match the executable bundle's bound; proposals still carry no binaries.
    metadata_limit_bytes: ClassVar[int | None] = 8 << 30
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
        # Copy external values into owned immutable records once per shared
        # Plan. Public to_dict() still returns independently editable values.
        entries = tuple(self.plans.items())  # Retain inputs until copying finishes.
        shared: dict[int, Any] = {}
        for _, value in entries:
            if id(value) not in shared:
                shared[id(value)] = freeze(value)
        object.__setattr__(
            self, "plans", freeze({key: shared[id(v)] for key, v in entries})
        )
        object.__setattr__(self, "rejected", freeze(self.rejected))

    def to_dict(self) -> dict[str, Any]:
        """Return the proposal as plain data.

        Returns
        -------
        dict
            ``version``, ``config``, ``profiles``, ``plans``, ``candidates``
            and ``rejected``.
        """
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
        """Rebuild a proposal from `to_dict` data.

        Parameters
        ----------
        value : dict
            The proposal, in `to_dict` form.
        payloads : mapping of str to bytes, optional
            Binary payloads; a proposal holds none, so it must be empty.

        Returns
        -------
        LocalPlans
            The proposal.

        Raises
        ------
        ValueError
            If ``payloads`` is not empty, or the record is not a version 1
            proposal.
        """
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
        """Save the proposal and return its reference for upload.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write.

        Returns
        -------
        ArtifactRef
            The file's reference, to pass as ``reference`` to
            `ArtifactTransfers.upload`.
        """
        self.save(path)
        with Path(path).open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        return ArtifactRef(
            "local-plans." + checksum,
            checksum,
            Path(path).stat().st_size,
            LOCAL_PLANS_MEDIA_TYPE,
        )
