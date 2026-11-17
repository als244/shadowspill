"""Self-contained prepared Plans, without live weights or hidden preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar

from shadowspill.artifacts._values import digest, freeze, plain
from shadowspill.artifacts.archive import Artifact
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.training.definitions.registry import Registry
from shadowspill_client._records import integer
from shadowspill_client.protocol.training import ResolvedTraining

from .bundle_manifest import manifest
from .family_records import PreparedPlans
from .identity import without_initialization
from .layouts import prepare_layouts
from .setup import resolve_training
from .templates import capture_inventory


@dataclass(frozen=True)
class TrainingBundle(Artifact):
    """An inspectable archive; ``recheck`` is required before worker creation.

    Raw loading verifies the archive and decodes CPU artifacts. Rechecking also
    requires the installed definitions and exact setup sources. Neither path
    constructs a model, initializes weights, profiles, or searches for a Plan.
    """

    artifact_kind: ClassVar[str] = "training_bundle"
    shared_metadata: ClassVar[bool] = True
    # A family retains complete schedules for multiple geometries/orderings.
    # Allow large real-model families; this is a corruption guard, not a budget.
    metadata_limit_bytes: ClassVar[int | None] = 8 << 30
    payload_limit_bytes: ClassVar[int | None] = 8 << 30
    config: ResolvedTraining
    sources: Mapping[str, ProfiledProgram]
    plans: Mapping[int, Mapping[str, Plan]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.config, ResolvedTraining):
            raise TypeError("bundle config must be ResolvedTraining")
        for key, source in self.sources.items():
            _identity(key)
            if not isinstance(source, ProfiledProgram):
                raise TypeError("bundle sources must be ProfiledPrograms")
        for rank, plans in self.plans.items():
            integer(rank, "logical rank")
            for key, plan in plans.items():
                _identity(key)
                if not isinstance(plan, Plan):
                    raise TypeError("bundle plans must be Plans")
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        object.__setattr__(
            self,
            "plans",
            MappingProxyType(
                {
                    rank: MappingProxyType(dict(plans))
                    for rank, plans in self.plans.items()
                }
            ),
        )
        object.__setattr__(self, "metadata", freeze(self.metadata))

    @classmethod
    def from_prepared(
        cls, prepared: PreparedPlans, sources: Mapping[str, ProfiledProgram]
    ) -> TrainingBundle:
        """Retain only the captures needed by the admitted capability family."""
        from .bundle_validation import recheck_bundle

        if sources.keys() != prepared.inventory.templates.keys():
            raise ValueError(
                "provide source profiles for exactly the capture inventory"
            )
        needed = {
            use.template
            for rows in prepared.families.values()
            for candidate in rows
            for worker in candidate.source.workers.values()
            for use in (worker.main, worker.optimizer)
            if use is not None
        }
        bundle = cls(
            prepared.inventory.layouts.setup.config,
            {key: sources[key] for key in sorted(needed)},
            prepared.plans,
            manifest(prepared),
        )
        recheck_bundle(bundle, prepared.inventory)
        return bundle

    def with_initialization(self, config: ResolvedTraining) -> TrainingBundle:
        """Retain executable Plans while selecting fresh weights or a checkpoint.

        The returned bundle requires the same strict ``recheck`` before use.
        Initialization does not change captures or admission; all other settings
        stay exact, apart from nonexecuting default/override provenance.
        """
        if without_initialization(config) != without_initialization(self.config):
            raise ValueError(
                "saved Plans differ from the resolved computation settings"
            )
        return replace(self, config=config)

    def recheck(self, registry: Registry) -> PreparedPlans:
        """Fail closed on missing/changed capabilities; never repair or replan."""
        from .bundle_validation import recheck_bundle

        setup = resolve_training(registry, config=self.config)
        return recheck_bundle(self, capture_inventory(prepare_layouts(setup)))

    def to_dict(self) -> dict[str, Any]:
        return self._value(checksums=True)

    def _archive_value(self) -> dict[str, Any]:
        # The archive checks the entire metadata table and every binary blob.
        # Nested checksums would serialize/hash the expanded shared tree again
        # on load. Source identities and Plan admission checks remain unchanged.
        return self._value(checksums=False)

    def _value(self, *, checksums: bool) -> dict[str, Any]:
        return {
            "version": 1 if checksums else 2,
            "config": self.config.to_dict(),
            "sources": [
                {"id": key, **_saved(source, checksums=checksums)}
                for key, source in sorted(self.sources.items())
            ],
            "plans": [
                {
                    "rank": rank,
                    "id": key,
                    **_saved(plan, checksums=checksums),
                }
                for rank, plans in sorted(self.plans.items())
                for key, plan in sorted(plans.items())
            ],
            "metadata": plain(self.metadata),
        }

    def _archive_payloads(self) -> Mapping[str, bytes]:
        return {
            key: value
            for artifact in (
                *self.sources.values(),
                *(p for ps in self.plans.values() for p in ps.values()),
            )
            for key, value in artifact._archive_payloads().items()
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> TrainingBundle:
        if (
            set(value) != {"version", "config", "sources", "plans", "metadata"}
            or type(value["version"]) is not int
            or value["version"] not in (1, 2)
        ):
            raise ValueError("unsupported training bundle schema")
        checksums = value["version"] == 1
        sources = {}
        for row in value["sources"]:
            _entry(row, {"id", "value"}, checksums=checksums)
            key = _identity(row["id"])
            if key in sources:
                raise ValueError("duplicate bundle source")
            sources[key] = ProfiledProgram.from_dict(row["value"], payloads=payloads)
        plans: dict[int, dict[str, Plan]] = {}
        for row in value["plans"]:
            _entry(row, {"rank", "id", "value"}, checksums=checksums)
            rank, key = integer(row["rank"], "logical rank"), _identity(row["id"])
            if key in plans.setdefault(rank, {}):
                raise ValueError("duplicate bundle Plan")
            plans[rank][key] = Plan.from_dict(row["value"], payloads=payloads)
        return cls(
            ResolvedTraining.from_dict(value["config"]),
            sources,
            plans,
            value["metadata"],
        )


def _identity(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("bundle identity must be a SHA-256 digest")
    return value


def _entry(row: Mapping[str, Any], keys: set[str], *, checksums: bool) -> None:
    if set(row) != (keys | {"digest"} if checksums else keys) or (
        checksums and row["digest"] != digest(row["value"])
    ):
        raise ValueError("bundle artifact identity differs from its saved value")


def _saved(artifact: Artifact, *, checksums: bool) -> dict[str, Any]:
    value = plain(artifact.to_dict()) if checksums else artifact.to_dict()
    return {"digest": digest(value), "value": value} if checksums else {"value": value}
