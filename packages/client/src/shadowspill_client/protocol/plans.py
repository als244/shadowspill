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
    "rejected_candidate",
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
        "rejected_candidate",
    }
)
MAX_SUMMARY_ROWS = 32
# Inspection rows now include immutable rich-view references. Old pages stay
# immutable when an existing bundle is prepared under this inspection schema.
SUMMARY_VERSION = 3
PLAN_REPORT_VERSION = 1
PLAN_REPORT_MEDIA_TYPE = "text/html; charset=utf-8"


@dataclass(frozen=True)
class PlanReportRequest(Record):
    session_id: str
    run_id: str
    plans: ArtifactRef
    max_bytes: int = 64 << 20

    def __post_init__(self) -> None:
        identifier(self.session_id, "report session ID")
        identifier(self.run_id, "report run ID")
        object.__setattr__(self, "plans", record(ArtifactRef, self.plans, "Plans"))
        integer(self.max_bytes, "report byte limit", minimum=1)
        if self.max_bytes > 1 << 30:
            raise ValueError("report byte limit exceeds 1 GiB")

    @property
    def report_id(self) -> str:
        identity = (
            f"{self.session_id}:{self.run_id}:{summary_id(self.plans, 0)}:"
            f"{PLAN_REPORT_VERSION}:{self.max_bytes}"
        )
        return "plan-report." + sha256(identity.encode()).hexdigest()


@dataclass(frozen=True)
class PlanReportDocument(Record):
    request: PlanReportRequest
    reference: ArtifactRef

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request", record(PlanReportRequest, self.request, "request")
        )
        object.__setattr__(
            self, "reference", record(ArtifactRef, self.reference, "report")
        )
        if (
            self.reference.id != self.request.report_id
            or self.reference.media_type != PLAN_REPORT_MEDIA_TYPE
            or self.reference.size_bytes > self.request.max_bytes
        ):
            raise ValueError("Plan report differs from its request")


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
    digest = sha256(identity.encode()).hexdigest()
    return f"plans-summary.v{SUMMARY_VERSION}.{digest}.{page}"


@dataclass(frozen=True)
class PlanSummaryRow(Record):
    """One fact from the metadata saved with a Plan bundle.

    Parameters
    ----------
    kind : str
        What the row describes:

        - ``"configuration"``: the model, objective, memory budgets, preset,
          engine and client sources, input grouping and the validation
          workspace of each operation.
        - ``"batch"``: one batch option of the run, with its index.
        - ``"build_case"``: one build case of the preparation, with its index.
        - ``"representative"``: the representative inputs of one build
          template.
        - ``"candidate"``: one packing candidate: its operation, batch
          option, build case, microbatch count, packing schedule, ordering,
          simulated seconds, scheduled capacities and the worker Plans it
          uses.
        - ``"worker_plan"``: one worker's Plan, with its simulated summary
          and declared spill bytes.
        - ``"rejected_candidate"``: a candidate preparation rejected, with
          the stage and reason.
        - ``"coverage"``: a geometry proof that batches within one batch
          option's limits pack into a named schedule of one operation.
        - ``"shared_memory"``: one worker's fixed slice, execution and spill
          memory, in bytes.
    values : mapping
        The row's fields, by name, as read-only JSON values.
    """

    kind: SummaryKind
    values: Mapping[str, Value]

    def __post_init__(self) -> None:
        if self.kind not in SUMMARY_KINDS:
            raise ValueError("unknown saved Plan summary row")
        object.__setattr__(self, "values", mapping(self.values, "Plan summary values"))


@dataclass(frozen=True)
class PlanSummaryPage(Record):
    """One page of the metadata saved with a Plan bundle.

    `PlansHandle.summary` returns it. Pages are immutable.

    Parameters
    ----------
    plans : ArtifactRef
        The Plan bundle described.
    configuration_sha256 : str
        SHA-256 digest of the resolved configuration the Plans were prepared
        for.
    page : int
        The page number, from 0.
    next_page : int or None
        Number of the next page, or None on the last page.
    candidate_count : int
        Number of packing candidates in the bundle, over every operation.
    plan_count : int
        Number of worker Plans in the bundle, over every worker.
    rows : sequence of PlanSummaryRow
        The page's rows, from 1 to 32.
    """

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
