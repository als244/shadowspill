"""Scalar history is a bounded projection of the authoritative request journal."""

from dataclasses import replace

from shadowspill_client.protocol.codec import ProtocolError, encode
from shadowspill_client.protocol.metrics import (
    MetricAnnotation,
    MetricPage,
    MetricRecord,
    metric_cursor,
)
from shadowspill_client.protocol.requests import OperationRequest
from shadowspill_client.protocol.results import RequestStatus

from .request_index import RequestIndex


def _metric(
    position: int, request: OperationRequest, status: RequestStatus
) -> MetricRecord:
    result, failure = status.result, status.failure
    if result is not None:
        return MetricRecord(
            position,
            request.identity,
            request.operation,
            result.outcome,
            result.history_id,
            result.completed_updates,
            result.state_valid,
            "completed",
            loss=result.loss,
            metrics=result.metrics,
        )
    if failure is None:
        raise ValueError("metric index names an unfinished request")
    return MetricRecord(
        position,
        request.identity,
        request.operation,
        "failed",
        failure.history_id,
        failure.completed_updates,
        failure.state_valid,
        failure.execution_effect,
        loss=failure.loss,
        metrics=failure.metrics,
        failure_code=failure.code,
    )


def caller_metric(
    position: int,
    annotation: MetricAnnotation,
    request: OperationRequest,
    status: RequestStatus,
) -> MetricRecord:
    if request.identity.request_id != annotation.request_id:
        raise ValueError("caller metrics name another request")
    if status.state not in ("completed", "failed"):
        raise ValueError("caller metrics require a terminal request result")
    return replace(
        _metric(position, request, status),
        source="caller",
        annotation_id=annotation.metric_id,
        loss=None,
        metrics=annotation.values,
    )


def metric_page(
    index: RequestIndex,
    *,
    session_id: str,
    run_id: str,
    position: int,
    after: int,
    through: int | None,
    limit: int,
) -> MetricPage:
    metric_cursor(after, "metric cursor")
    end = position if through is None else metric_cursor(through, "metric snapshot")
    if not after <= end <= position or type(limit) is not int or not 1 <= limit <= 128:
        raise ValueError("metrics need a valid journal range and limit 1..128")
    candidates = index.metric_events(after, end, limit + 1)
    rows: list[MetricRecord] = []
    page = MetricPage(session_id, run_id, after, end, ())
    for number, (cursor, request_id, annotation_id) in enumerate(candidates[:limit]):
        if annotation_id is None:
            source = index.get(request_id)
            if source is None:
                raise ValueError("metric index lost its authoritative request")
            row = _metric(cursor, source[0], source[1])
            del source
        else:
            annotation = index.annotation(request_id, annotation_id)
            if annotation is None or annotation.position != cursor:
                raise ValueError("metric index lost its authoritative annotation")
            row = annotation
        candidate = MetricPage(
            session_id,
            run_id,
            after,
            end,
            (*rows, row),
            cursor if number + 1 < len(candidates) else None,
        )
        try:
            encode(candidate)
        except ProtocolError:
            if not rows:
                raise
            break
        rows.append(row)
        page = candidate
    return page
