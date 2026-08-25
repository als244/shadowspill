"""Reading one diagnostic record back out of JSON, one field at a time."""

from __future__ import annotations


def _parse_candidate_id(value: str) -> tuple[str, str, bool]:
    coalesced = value.endswith("-coalesced")
    base = value[: -len("-coalesced")] if coalesced else value
    strategy, separator, rule = base.partition("/")
    if not separator:
        return "unknown", "unknown", coalesced
    return strategy, rule, coalesced


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object")
    return value


def without_measurements(value: object) -> object:
    """Strip everything that measures the run rather than describing the plan.

    Two runs of the same input produce the same plan and different timings, so
    anything compared or digested across runs has to leave the timings out.
    `sections` goes whole: every number in it is a measurement.
    """

    # Tuples become lists: a payload read back from JSON has lists where the
    # payload just built has tuples, and they have to compare equal.
    if isinstance(value, (list, tuple)):
        return [without_measurements(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: without_measurements(item)
        for key, item in value.items()
        if key != "sections" and not key.endswith("_time_ns")
    }


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _optional_integer(value: object, path: str) -> int | None:
    return None if value is None else _integer(value, path)


def _optional_string(value: object, path: str) -> str | None:
    return None if value is None else _string(value, path)
