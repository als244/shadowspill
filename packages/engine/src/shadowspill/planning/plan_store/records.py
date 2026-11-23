"""Planning-cache results and format identifiers."""

import dataclasses
import hashlib
import json
import sys
import typing
from collections.abc import Iterator
from dataclasses import dataclass

from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.memory_search.admission.layout.model import (
    FixedLayoutAdmission,
)
from shadowspill.planning.memory_search.result import ProgramPlanResult
from shadowspill.schema import artifact_schema

SCHEMA = artifact_schema("plan_selection")
RESOLUTION_SCHEMA = artifact_schema("plan_resolution")


def _types(hint: object) -> Iterator[object]:
    yield hint
    for argument in typing.get_args(hint):
        yield from _types(argument)


def record_layout(root: type) -> str:
    """Digest the field names of a record type and of every record it holds.

    A cached record keyed on it is never read by code that records something
    else: adding, removing or renaming a field anywhere in the record changes
    the digest. An annotation that names a type imported only for type
    checking is not followed; the key covers such values on its own.
    """

    layout = {}
    pending = [root]
    while pending:
        record = pending.pop()
        name = f"{record.__module__}.{record.__qualname__}"
        if name in layout:
            continue
        fields = dataclasses.fields(record)
        layout[name] = [field.name for field in fields]
        namespace = vars(sys.modules[record.__module__])
        for field in fields:
            try:
                hint = (
                    eval(field.type, namespace)
                    if isinstance(field.type, str)
                    else field.type
                )
            except NameError:
                continue
            pending.extend(
                item
                for item in _types(hint)
                if isinstance(item, type) and dataclasses.is_dataclass(item)
            )
    encoded = json.dumps(layout, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


#: The layout of what a cached selection and its resolutions record.
SELECTION_LAYOUT = record_layout(ProgramPlanResult)


@dataclass(frozen=True, slots=True)
class PlanLookup:
    """Selected result and its optional fixed-layout certificate."""

    result: ProgramPlanResult
    from_store: bool
    key: str = ""
    certificate: FixedLayoutAdmission | None = None


@dataclass(frozen=True, slots=True)
class Verdict:
    """A cached search refusal; environment/runtime failures are not cached."""

    outcome: str
    error: str
    kind: str | None
    message: str

    def exception(self) -> Exception:
        if self.outcome == "exhausted":
            return PlanSearchExhaustedError(self.message)
        return PlanInfeasibleError(
            self.message, kind=self.kind if self.kind is not None else self.error
        )
