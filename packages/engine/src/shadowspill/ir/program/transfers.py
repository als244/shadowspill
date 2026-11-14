"""Physical spill sizes and ordered transfer phases for one logical alias.

Copy phases use the calibrated directional host link. Timed phases use an
explicit serial resource, for example a collective communicator. Readiness is
published only after the entire recipe completes. Runtime adapters must provide
the same transformations; a timing recipe alone does not implement a transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..serialization import JsonValue
from ..validation import (
    expect_boolean,
    expect_integer,
    expect_list,
    expect_mapping,
    expect_string,
    field,
    require,
    require_identifier,
    require_non_negative,
    require_tuple,
)


@dataclass(frozen=True, slots=True)
class TransferPhase:
    kind: str
    bytes: int = 0
    duration_ns: int = 0
    resource: str | None = None
    submission_ordered: bool = False
    timing_class: str | None = None

    def __post_init__(self) -> None:
        if self.timing_class is not None:
            require_identifier(self.timing_class, "transfer.timing_class")
            require(
                self.kind == "timed", "transfer.timing_class", "requires timed work"
            )
        require(self.kind in {"copy", "timed"}, "transfer.kind", "unknown phase")
        require_non_negative(self.bytes, "transfer.bytes")
        require_non_negative(self.duration_ns, "transfer.duration_ns")
        require(
            isinstance(self.submission_ordered, bool),
            "transfer.submission_ordered",
            "expected bool",
        )
        if self.kind == "copy":
            require(self.resource is None, "transfer.resource", "copy uses its lane")
            require(self.duration_ns == 0, "transfer.duration_ns", "copy uses rates")
            require(
                not self.submission_ordered,
                "transfer.submission_ordered",
                "only timed phases are ordered",
            )
        else:
            require(
                self.resource is not None, "transfer.resource", "resource is required"
            )
            assert self.resource is not None
            require_identifier(self.resource, "transfer.resource")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "bytes": self.bytes,
            "duration_ns": self.duration_ns,
            "resource": self.resource,
            "submission_ordered": self.submission_ordered,
            "timing_class": self.timing_class,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TransferPhase:
        data = expect_mapping(value, path)
        resource = field(data, "resource", path)
        return cls(
            expect_string(field(data, "kind", path), f"{path}.kind"),
            expect_integer(field(data, "bytes", path), f"{path}.bytes"),
            expect_integer(field(data, "duration_ns", path), f"{path}.duration_ns"),
            None if resource is None else expect_string(resource, f"{path}.resource"),
            expect_boolean(
                data.get("submission_ordered", False), f"{path}.submission_ordered"
            ),
            None
            if data.get("timing_class") is None
            else expect_string(data["timing_class"], f"{path}.timing_class"),
        )


@dataclass(frozen=True, slots=True)
class TransferSpec:
    spill_size_bytes: int
    fetch: tuple[TransferPhase, ...]
    evict: tuple[TransferPhase, ...]

    def __post_init__(self) -> None:
        require_non_negative(self.spill_size_bytes, "transfer.spill_size_bytes")
        for name in ("fetch", "evict"):
            phases = getattr(self, name)
            require_tuple(phases, f"transfer.{name}")
            require(bool(phases), f"transfer.{name}", "recipe must not be empty")
            require(
                all(isinstance(phase, TransferPhase) for phase in phases),
                f"transfer.{name}",
                "invalid phase",
            )
            require(
                sum(phase.submission_ordered for phase in phases) <= 1,
                f"transfer.{name}",
                "one ordered collective phase per transfer",
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "spill_size_bytes": self.spill_size_bytes,
            "fetch": [phase.to_dict() for phase in self.fetch],
            "evict": [phase.to_dict() for phase in self.evict],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TransferSpec:
        data = expect_mapping(value, path)

        def phases(name: str) -> tuple[TransferPhase, ...]:
            return tuple(
                TransferPhase.from_value(item, f"{path}.{name}[{index}]")
                for index, item in enumerate(
                    expect_list(field(data, name, path), f"{path}.{name}")
                )
            )

        return cls(
            expect_integer(
                field(data, "spill_size_bytes", path), f"{path}.spill_size_bytes"
            ),
            phases("fetch"),
            phases("evict"),
        )
