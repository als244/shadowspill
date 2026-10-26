"""Attach saved task identity to preparation failures without hiding their cause."""

from shadowspill.artifacts.program import Program
from shadowspill.errors import TaskPhaseError


def task_error(
    error_type: type[TaskPhaseError],
    program: Program,
    key: str,
    phase: str,
    cause: Exception,
) -> TaskPhaseError:
    definition = next(t for t in program.task_definitions if t.profile_key == key)
    metadata = program.bindings.get("task_metadata", {}).get(key, {})
    return error_type(
        f"{phase} failed for {definition.name or key} ({key}): {cause}",
        structural_contract=metadata.get("structural_contract", key),
        task_kind=metadata.get("kind"),
        operators=tuple(metadata.get("operators", ())),
    )
