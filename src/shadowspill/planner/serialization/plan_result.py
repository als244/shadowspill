"""Reading back the parts of a plan result that are not their own types."""

from __future__ import annotations

from shadowspill.planner.request import GenericPlanningOptions
from shadowspill.planner.result import ResidentSlice

from .common import (
    _integer,
    _list,
    _mapping,
    _string,
)


def _options_from_value(value: object, path: str) -> GenericPlanningOptions:
    data = _mapping(value, path)
    try:
        return GenericPlanningOptions.from_dict(data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} is invalid: {error}") from error


def _resident_slice_from_value(value: object, path: str) -> ResidentSlice:
    data = _mapping(value, path)
    entries = _list(data.get("aliases"), f"{path}.aliases")
    return ResidentSlice(
        bytes=_integer(data.get("bytes"), f"{path}.bytes"),
        aliases=tuple(
            _string(item, f"{path}.aliases[{index}]")
            for index, item in enumerate(entries)
        ),
    )

