"""Bounded saved-Plan inspection, without importing executable Plan objects."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from .._records import Record, Value, integer, mapping, record, text
from .requests import ArtifactRef, identifier

type SummaryKind = Literal[
    "configuration",
    "batch",
    "build_case",
    "candidate",
    "worker_plan",
    "coverage",
    "shared_memory",
    "representative",
]
SUMMARY_KINDS = frozenset(
    {
        "configuration",
        "batch",
        "build_case",
        "candidate",
        "worker_plan",
        "coverage",
        "shared_memory",
        "representative",
    }
)
MAX_SUMMARY_ROWS = 32


def plan_name(name: str) -> str:
    if len(text(name, "Plan name")) > 256:
        raise ValueError("Plan name exceeds its bound")
    name.encode("utf-8")  # Reject lone surrogates before durable storage or a URL.
    return name


@dataclass(frozen=True)
class SavedPlans(Record):
    """A run-local name for immutable server Plans, independent of workers."""

    session_id: str
    run_id: str
    name: str
    plans: ArtifactRef

    def __post_init__(self) -> None:
        identifier(self.session_id, "Plan session ID")
        identifier(self.run_id, "Plan run ID")
        plan_name(self.name)
        object.__setattr__(self, "plans", record(ArtifactRef, self.plans, "Plans"))


def summary_id(plans: ArtifactRef, page: int) -> str:
    integer(page, "summary page")
    # Bind both the immutable content and its owner-scoped name. A second name
    # for identical bytes must not overwrite a page naming the first artifact.
    identity = "\0".join(
        (plans.id, plans.sha256, str(plans.size_bytes), plans.media_type)
    )
    return f"plans-summary.{sha256(identity.encode()).hexdigest()}.{page}"


@dataclass(frozen=True)
class PlanSummaryRow(Record):
    kind: SummaryKind
    values: Mapping[str, Value]

    def __post_init__(self) -> None:
        if self.kind not in SUMMARY_KINDS:
            raise ValueError("unknown saved Plan summary row")
        object.__setattr__(self, "values", mapping(self.values, "Plan summary values"))


@dataclass(frozen=True)
class PlanSummaryPage(Record):
    plans: ArtifactRef
    configuration_sha256: str
    page: int
    next_page: int | None
    candidate_count: int
    plan_count: int
    rows: Sequence[PlanSummaryRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "plans", record(ArtifactRef, self.plans, "Plans"))
        ArtifactRef("configuration", self.configuration_sha256, 0)
        for name in ("page", "candidate_count", "plan_count"):
            integer(getattr(self, name), name)
        if self.next_page is not None:
            integer(self.next_page, "next summary page")
            if self.next_page != self.page + 1:
                raise ValueError("summary pages must be contiguous")
        if (
            not isinstance(self.rows, (list, tuple))
            or not 1 <= len(self.rows) <= MAX_SUMMARY_ROWS
        ):
            raise ValueError("summary page requires 1-32 rows")
        object.__setattr__(
            self,
            "rows",
            tuple(record(PlanSummaryRow, row, "summary row") for row in self.rows),
        )
