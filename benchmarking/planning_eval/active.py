"""The currently running point, used to recover interrupted workers."""

from pathlib import Path

from benchmarking._serialization import atomic_json
from benchmarking.journals import read_object, utc_now
from shadowspill.schema import artifact_schema

from .matrix import FrontierPointRequest


def write_active_point(
    case_directory: Path,
    request: FrontierPointRequest | None,
) -> None:
    atomic_json(
        case_directory / "active.json",
        {
            "schema": artifact_schema("search_frontier_active"),
            "updated_at": utc_now(),
            "point_id": None if request is None else request.point_id,
            "request_digest": None if request is None else request.digest,
        },
    )


def read_active_point(case_directory: Path) -> tuple[str, str] | None:
    path = case_directory / "active.json"
    if not path.exists():
        return None
    value = read_object(path)
    point_id = value.get("point_id")
    request_digest = value.get("request_digest")
    if point_id is None and request_digest is None:
        return None
    if not isinstance(point_id, str) or not isinstance(request_digest, str):
        raise ValueError(f"invalid active point journal at {path}")
    return point_id, request_digest
