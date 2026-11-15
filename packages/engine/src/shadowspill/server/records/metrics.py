"""Scalar history is a bounded projection of the authoritative request journal."""

from shadowspill_client.protocol.codec import ProtocolError, encode
from shadowspill_client.protocol.metrics import MetricPage, MetricRecord, metric_cursor
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
        failure_code=failure.code,
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
    candidates = index.finished(after, end, limit + 1)
    rows: list[MetricRecord] = []
    page = MetricPage(session_id, run_id, after, end, ())
    for number, (cursor, request_id) in enumerate(candidates[:limit]):
        source = index.get(request_id)
        if source is None:
            raise ValueError("metric index lost its authoritative request")
        row = _metric(cursor, source[0], source[1])
        del source
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
