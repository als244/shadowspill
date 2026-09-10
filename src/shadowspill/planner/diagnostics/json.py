"""Reading one diagnostic record back out of JSON, one field at a time.

The strict readers themselves are `planner.strict`'s: one
malformed record has to refuse in the same words whichever half of the
planner parsed it.
"""

from __future__ import annotations

from ..strict import (
    _boolean,
    _integer,
    _list,
    _mapping,
    _optional_integer,
    _optional_string,
    _string,
)

__all__ = [
    "_boolean",
    "_integer",
    "_list",
    "_mapping",
    "_optional_integer",
    "_optional_string",
    "_parse_candidate_id",
    "_span",
    "_string",
    "without_measurements",
]


def _parse_candidate_id(value: str) -> tuple[str, str, bool]:
    coalesced = value.endswith("-coalesced")
    base = value[: -len("-coalesced")] if coalesced else value
    strategy, separator, rule = base.partition("/")
    if not separator:
        return "unknown", "unknown", coalesced
    return strategy, rule, coalesced


def without_measurements(value: object) -> object:
    """Strip everything that measures the run rather than describing the plan.

    Two runs of the same input produce the same plan and different timings, so
    anything compared or digested across runs has to leave the timings out.
    `sections` and `span` go whole: every number in them is a measurement.
    `workers` goes for the same reason from the other direction: it is how
    much machine was spent, not what was decided, and two runs at different
    worker counts describe the same plan.
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
        if key not in ("sections", "span", "workers")
        and not key.endswith("_time_ns")
    }


def _span(value: object, name: str, path: str) -> int:
    """One end of a wall-clock span, absent from anything digested or old."""

    if value is None:
        return 0
    return _optional_integer(_mapping(value, path).get(name), f"{path}.{name}") or 0
