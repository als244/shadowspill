"""Publish bounded immutable inspection pages while the creation owns its Plans."""

import hashlib
import io
from collections.abc import Iterable
from dataclasses import replace

from shadowspill.diagnostics.occupancy.page import script_json
from shadowspill.reporting.occupancy import occupancy_data
from shadowspill.training.preparation.artifacts.inspection import summary_rows
from shadowspill.training.preparation.planning.records import PreparedPlans
from shadowspill_client._records import Value, mapping
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
        _inspection_rows(artifacts, prepared),
        configuration_sha256=digest(prepared.inventory.layouts.setup.config),
        candidate_count=sum(len(rows) for rows in prepared.families.values()),
        plan_count=sum(len(plans) for plans in prepared.plans.values()),
    )


def _inspection_rows(
    artifacts: ArtifactStore, prepared: PreparedPlans
) -> Iterable[PlanSummaryRow]:
    for row in summary_rows(prepared):
        if row.kind == "worker_plan":
            rank, identity = row.values["rank"], row.values["id"]
            assert isinstance(rank, int) and isinstance(identity, str)
            plan = prepared.plans[rank][identity]
            views: list[Value] = []
            paths = (("fresh", plan), ("accumulated", plan._accumulation_plan))
            for mode, path in paths:
                if path is None:
                    continue
                payload = script_json(
                    occupancy_data(
                        path, description=f"Rank {rank} · {identity} · {mode}"
                    )
                ).encode()
                if len(payload) > 16 << 20:
                    raise ValueError("one Plan inspection view exceeds 16 MiB")
                checksum = hashlib.sha256(payload).hexdigest()
                reference = ArtifactRef(
                    "plan-view." + checksum, checksum, len(payload), "application/json"
                )

                def open_payload(_: ArtifactRef, data: bytes = payload) -> io.BytesIO:
                    return io.BytesIO(data)

                artifacts.put(reference, open_payload)
                views.append(
                    mapping({"mode": mode, "data": reference.to_dict()}, "Plan view")
                )
            row = replace(row, values={**row.values, "views": tuple(views)})
        yield row


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
