"""Attach saved task identity to preparation failures without hiding their cause."""

from shadowspill.artifacts.program import Program
from shadowspill.artifacts.task_names import profile_labels
from shadowspill.errors import TaskPhaseError


def task_error(
    error_type: type[TaskPhaseError],
    program: Program,
    key: str,
    phase: str,
    cause: Exception,
) -> TaskPhaseError:
    metadata = program.bindings.get("task_metadata", {}).get(key, {})
    return error_type(
        f"{phase} failed for {profile_labels(program).get(key, key)} ({key}): {cause}",
        structural_contract=metadata.get("structural_contract", key),
        task_kind=metadata.get("kind"),
        operators=tuple(metadata.get("operators", ())),
    )
