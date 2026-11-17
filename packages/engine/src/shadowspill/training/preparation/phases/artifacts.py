"""Resume exact phase artifacts and classify only existing recoverable failures."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.artifacts.program import Program
from shadowspill.communication.info import CommunicationInfo
from shadowspill.orchestration.launch.workers import Workers
from shadowspill.training.preparation.capture.allocation_coverage import (
    validate_allocation_coverage,
)
from shadowspill.training.preparation.capture.failures import (
    PreparationFailure,
    prune_failed_templates,
)
from shadowspill.training.preparation.capture.target import check_target
from shadowspill.training.preparation.capture.templates import (
    TemplateInventory,
    validate_template,
)
from shadowspill.training.preparation.planning.identity import validate_profile
from shadowspill.training.preparation.planning.records import PlanRejection


def attempt(
    root: Path,
    key: str,
    phase: Literal["build", "profile"],
    operation: Callable[[], None],
    group: Workers,
) -> bool:
    """A collective failure aborts its workers; only independent OOMs are pruned."""
    if group.count > 1:
        operation()
        return True
    from shadowspill.orchestration.candidates import attempt as try_candidate

    entry: dict[str, Any] = {"name": key}
    try_candidate(entry, phase, operation, device=0)
    if "failure" in entry:
        PreparationFailure(key, phase, entry["failure"]["message"]).save(
            root / "failures" / f"{key}.ssp"
        )
    else:
        (root / "failures" / f"{key}.ssp").unlink(missing_ok=True)
    return "failure" not in entry


def reused[T](path: Path, group: Workers, load: Callable[[], T]) -> T | None:
    """All participants take the same branch and report validation failures."""
    if not all(group.gather(path.is_file())):
        return None
    result, error = None, None
    try:
        result = load()
    except Exception as failure:
        error = str(failure)
    errors = [message for message in group.gather(error) if message is not None]
    if errors:
        raise ValueError(
            f"cannot reuse saved phase artifact {path}: {'; '.join(errors)}"
        )
    print(f"reusing {path}", flush=True)
    return result


def program_at(root: Path, inventory: TemplateInventory, key: str) -> Program:
    program = Program.load(root / "programs" / f"{key}.ssp")
    validate_template(inventory.templates[key], program)
    validate_allocation_coverage(inventory.templates[key], program)
    check_target(program)
    return program


def profile_at(root: Path, inventory: TemplateInventory, key: str) -> ProfiledProgram:
    profiled = ProfiledProgram.load(root / "profiled" / f"{key}.ssp")
    validate_profile(inventory, key, profiled)
    check_target(profiled.program)
    source = program_at(root, inventory, key)
    if profiled.program.digest != source.digest:
        raise ValueError("saved profile belongs to another source capture")
    return profiled


def calibration_at(root: Path, inventory: TemplateInventory, key: str) -> bool:
    """Retained overlap/communication must preserve the original memory evidence."""
    source = profile_at(root, inventory, key)
    calibrated = ProfiledProgram.load(root / "calibrated" / f"{key}.ssp")
    communication = CommunicationInfo.load(root / "communication" / f"{key}.ssp")
    validate_profile(inventory, key, calibrated)
    if (
        calibrated.program.digest != source.program.digest
        or calibrated.fixed_device_bytes != source.fixed_device_bytes
        or communication.metadata.get("profiling_options")
        != inventory.layouts.setup.config.preparation.profiling.to_dict()
    ):
        raise ValueError("saved calibration differs from its measured capture/policy")
    for key, original in source.task_profiles.items():
        actual = calibrated.task_profiles[key]
        if any(
            getattr(actual, field) != getattr(original, field)
            for field in (
                "duration_seconds",
                "workspace_bytes",
                "compatibility_digest",
                "resource_phases",
            )
        ) or any(
            actual.metadata.get(name) != value
            for name, value in original.metadata.items()
        ):
            raise ValueError("saved overlap calibration changed original task evidence")
    return True


def admitted_inventory(
    root: Path, inventory: TemplateInventory
) -> tuple[TemplateInventory, tuple[PlanRejection, ...]]:
    failures = {
        key: PreparationFailure.load(path)
        for key in inventory.templates
        if (path := root / "failures" / f"{key}.ssp").is_file()
    }
    return prune_failed_templates(inventory, failures)
