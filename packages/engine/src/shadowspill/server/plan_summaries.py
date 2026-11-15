"""Publish bounded immutable inspection pages while the creation owns its Plans."""

import hashlib
import io
from collections.abc import Iterable

from shadowspill.training.preparation.family_records import PreparedPlans
from shadowspill.training.preparation.inspection import summary_rows
from shadowspill_client.protocol.codec import ProtocolError, digest, encode
from shadowspill_client.protocol.plans import (
    MAX_SUMMARY_ROWS,
    PlanSummaryPage,
    PlanSummaryRow,
    summary_id,
)
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.service import CONTROL_MEDIA_TYPE

from .artifacts import ArtifactStore


def publish_summary(
    artifacts: ArtifactStore, plans: ArtifactRef, prepared: PreparedPlans
) -> None:
    _publish_pages(
        artifacts,
        plans,
        summary_rows(prepared),
        configuration_sha256=digest(prepared.inventory.layouts.setup.config),
        candidate_count=sum(len(rows) for rows in prepared.families.values()),
        plan_count=sum(len(plans) for plans in prepared.plans.values()),
    )


def _publish_pages(
    artifacts: ArtifactStore,
    plans: ArtifactRef,
    rows: Iterable[PlanSummaryRow],
    *,
    configuration_sha256: str,
    candidate_count: int,
    plan_count: int,
) -> None:
    pending: list[PlanSummaryRow] = []
    page = 0

    def encoded(values: list[PlanSummaryRow], *, last: bool = False) -> bytes:
        return encode(
            PlanSummaryPage(
                plans,
                configuration_sha256,
                page,
                None if last else page + 1,
                candidate_count,
                plan_count,
                values,
            )
        )

    def publish(payload: bytes) -> None:
        reference = ArtifactRef(
            summary_id(plans, page),
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            CONTROL_MEDIA_TYPE,
        )
        artifacts.put(reference, lambda _: io.BytesIO(payload))

    for row in rows:
        # Preflight a single row before publishing any preceding page; never
        # truncate a row to hide an exceeded shared metadata bound.
        encoded([row])
        if pending:
            try:
                if len(pending) >= MAX_SUMMARY_ROWS:
                    raise ProtocolError("summary page row limit")
                encoded([*pending, row])
            except ProtocolError:
                publish(encoded(pending))
                page += 1
                pending = []
        pending.append(row)
    if not pending:
        raise ValueError("saved Plan inspection requires metadata")
    publish(encoded(pending, last=True))
