"""Explicit bounded reports from a fixed range of retained results."""

from dataclasses import dataclass
from typing import Literal

from .._records import Record, integer, record
from .metrics import MetricSnapshot, metric_cursor
from .requests import ArtifactRef

REPORT_MEDIA = {"html": "text/html; charset=utf-8", "json": "application/json"}
# Increment when rendered bytes or schema change; stored reports are immutable.
REPORT_VERSION = 3


@dataclass(frozen=True)
class ReportSettings(Record):
    """What a report of a run's metric journal covers, and its limits.

    `TrainingClient.report` sends its arguments as one.

    Parameters
    ----------
    format : str, default "html"
        ``"html"`` or ``"json"``.
    after : int, default 0
        Journal position after which records are included.
    through : int, optional
        Last journal position included, at least ``after``. None for the
        journal's end when the service makes the report; a `ReportDocument`
        holds the position it resolved to.
    max_records : int, default 10000
        Maximum number of records the report may hold, from 1 to 100000.
    max_bytes : int, default 64 << 20
        Largest report size, in bytes (64 MiB by default), up to 1 GiB.
    """

    format: Literal["html", "json"] = "html"
    after: int = 0
    through: int | None = None
    max_records: int = 10000
    max_bytes: int = 64 << 20

    def __post_init__(self) -> None:
        if self.format not in REPORT_MEDIA:
            raise ValueError("report format must be html or json")
        metric_cursor(self.after, "report start")
        if (
            self.through is not None
            and metric_cursor(self.through, "report end") < self.after
        ):
            raise ValueError("report start exceeds its snapshot")
        if integer(self.max_records, "report record limit", minimum=1) > 100000:
            raise ValueError("report record limit exceeds 100000")
        if integer(self.max_bytes, "report byte limit", minimum=1) > 1 << 30:
            raise ValueError("report byte limit exceeds 1 GiB")


@dataclass(frozen=True)
class ReportDocument(Record):
    """A stored report and the journal range it covers.

    `ReportHandle.document` holds it.

    Parameters
    ----------
    snapshot : MetricSnapshot
        The run and the journal range the report covers.
    settings : ReportSettings
        The settings the report was made with, ``through`` resolved to
        ``snapshot.through``.
    reference : ArtifactRef
        The stored report: HTML (``text/html; charset=utf-8``) or JSON
        (``application/json``), at most ``settings.max_bytes``.
    plans : ArtifactRef, optional
        The run's execution Plans; None when the service records none.
    version : int, default 3
        Version of the report's content and schema; the client accepts only
        the current one, 3.
    """

    snapshot: MetricSnapshot
    settings: ReportSettings
    reference: ArtifactRef
    plans: ArtifactRef | None = None
    version: int = REPORT_VERSION

    def __post_init__(self) -> None:
        for name, kind in (
            ("snapshot", MetricSnapshot),
            ("settings", ReportSettings),
            ("reference", ArtifactRef),
        ):
            object.__setattr__(self, name, record(kind, getattr(self, name), name))
        if self.plans is not None:
            object.__setattr__(
                self, "plans", record(ArtifactRef, self.plans, "report Plans")
            )
        if type(self.version) is not int or self.version != REPORT_VERSION:
            raise ValueError("unsupported report version")
        if (
            self.settings.after != self.snapshot.after
            or self.settings.through != self.snapshot.through
            or self.reference.size_bytes > self.settings.max_bytes
            or self.reference.media_type != REPORT_MEDIA[self.settings.format]
        ):
            raise ValueError("report reference differs from its snapshot/settings")
