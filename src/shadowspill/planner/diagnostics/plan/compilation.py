"""What building the plan cost: phases, the compiler, the caches it touched."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlanPhaseTiming:
    """One non-overlapping interval measured during frontend planning."""

    name: str
    duration_ns: int

    @property
    def duration_seconds(self) -> float:
        return self.duration_ns / 1e9

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "duration_ns": self.duration_ns,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class PlanCompilerProfile:
    """Non-overlapping compiler phases for one structural task contract."""

    structural_contract_key: str
    phases: tuple[PlanPhaseTiming, ...]

    @property
    def total_wall_time_ns(self) -> int:
        return sum(item.duration_ns for item in self.phases)

    def as_dict(self) -> dict[str, object]:
        return {
            "structural_contract_key": self.structural_contract_key,
            "phases": [item.as_dict() for item in self.phases],
            "total_wall_time_ns": self.total_wall_time_ns,
        }


@dataclass(frozen=True, slots=True)
class PlanCacheArtifact:
    """One persistent planning artifact touched by this planning call.

    ``access`` distinguishes bytes actually read or written from an existing
    artifact that merely matched a freshly produced in-memory result.  the framework
    the compiler's implementation-private directory is reported as ``managed``.
    """

    category: str
    kind: str
    digest: str | None
    path: str
    access: str
    schema: str | None = None
    dependencies: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "kind": self.kind,
            "digest": self.digest,
            "path": self.path,
            "access": self.access,
            "schema": self.schema,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True, slots=True)
class PlanProfilingMetadata:
    """Canonical planning-only workload metadata for one input position."""

    position: int
    digest: str
    canonical_json: str

    def as_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "digest": self.digest,
            "canonical_json": self.canonical_json,
        }
