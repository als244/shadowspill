"""Derive the collection summary from per-program journals."""

from collections.abc import Iterable

from benchmarking._serialization import atomic_json
from benchmarking.journals import utc_now

from .journal import load_case_status
from .matrix import ProgramRequest
from .paths import CollectionPaths

_SUMMARY_SCHEMA = "shadowspill.program_corpus_collection.summary/v1"


def write_summary(
    paths: CollectionPaths,
    requests: Iterable[ProgramRequest],
    *,
    started_at: str,
    selected_case_ids: Iterable[str],
) -> dict[str, object]:
    """Reconcile and persist aggregate status after every case."""

    counts = {
        name: 0 for name in ("pending", "running", "succeeded", "failed", "interrupted")
    }
    failed_cases: list[str] = []
    all_requests = tuple(requests)
    for request in all_requests:
        status = str(load_case_status(paths, request).get("status", "pending"))
        if status not in counts:
            status = "failed"
        counts[status] += 1
        if status in {"failed", "interrupted"}:
            failed_cases.append(request.case_id)
    summary: dict[str, object] = {
        "schema": _SUMMARY_SCHEMA,
        "collection_name": all_requests[0].collection_name,
        "config_digest": all_requests[0].config_digest,
        "started_at": started_at,
        "updated_at": utc_now(),
        "total_programs": len(all_requests),
        "selected_case_ids": list(selected_case_ids),
        "counts": counts,
        "failed_cases": failed_cases,
    }
    atomic_json(paths.summary_path, summary)
    return summary
