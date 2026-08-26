"""Immutable storage for annotated planner outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from benchmarking._serialization import (
    canonical_text_digest,
    commit_immutable_directory,
    existing_artifact_is_identical,
    json_mapping,
    mapping,
    read_mapping,
    string,
    text_digest,
)
from benchmarking.program_collection.corpus import (
    SavedProgramCase,
    load_step_program,
)
from shadowspill.pytorch import AnnotatedProgramPlan, StepProgram

_SELECTION_SCHEMA = "shadowspill.planning_corpus.selection/v1"


def save_annotated_plan(
    case: SavedProgramCase,
    plan: AnnotatedProgramPlan,
    *,
    metadata: Mapping[str, object] | None = None,
    step_program: StepProgram | None = None,
    output_root: Path,
) -> Path:
    """Save one plan under independent budget and bandwidth axes."""

    if step_program is None:
        saved_case, selected_program = load_step_program(case.directory)
        if saved_case != case:
            raise ValueError("saved Program case identity changed on disk")
    else:
        selected_program = step_program
        if selected_program.digest != case.program_digest:
            raise ValueError("provided StepProgram does not match the saved case")
    program_digests = {selected_program.recurrent.program.digest}
    if selected_program.initial is not None:
        program_digests.add(selected_program.initial.program.digest)
    if plan.program.program.digest not in program_digests:
        raise ValueError("annotated plan does not belong to the saved StepProgram")

    budgets = plan.memory_budgets
    bandwidths = plan.transfer_bandwidths
    # `to_json` is canonical by construction, so it is both what we store and
    # what identifies the artifact -- no second serialisation of either.
    payload = plan.to_json()
    artifact_digest = canonical_text_digest(payload)
    directory = (
        output_root.expanduser().resolve()
        / f"execution-{budgets.execution_bytes}_spill-{budgets.spill_bytes}"
        / (
            f"fetch-{bandwidths.fetch_bytes_per_second}_"
            f"evict-{bandwidths.evict_bytes_per_second}"
        )
        / plan.digest
        / f"artifact-{artifact_digest}"
    )
    manifest = {
        "schema": _SELECTION_SCHEMA,
        "source_step_program_digest": case.program_digest,
        "source_pressurefit_program_digest": plan.program.digest,
        "memory_budgets": plan.memory_budgets.to_dict(),
        "transfer_bandwidths": plan.transfer_bandwidths.to_dict(),
        "annotated_program_plan": {
            "path": "annotated_program_plan.json",
            "plan_digest": plan.digest,
            "artifact_sha256": artifact_digest,
        },
        "metadata": json_mapping(metadata or {}, "metadata"),
    }
    plan_path = directory / "annotated_program_plan.json"
    manifest_path = directory / "manifest.json"
    if not existing_artifact_is_identical(
        plan_path,
        manifest_path,
        payload=payload,
        manifest=manifest,
    ):
        commit_immutable_directory(
            directory,
            artifact_name="annotated_program_plan.json",
            payload=payload,
            manifest=manifest,
        )
    return directory.resolve()


def load_annotated_plan(path: Path) -> AnnotatedProgramPlan:
    """Load and integrity-check an annotated-plan directory or JSON file."""

    directory = path if path.is_dir() else path.parent
    manifest = read_mapping(directory / "manifest.json", "selection manifest")
    if manifest.get("schema") != _SELECTION_SCHEMA:
        raise ValueError("selection manifest has an unsupported schema")
    artifact = mapping(
        manifest.get("annotated_program_plan"),
        "selection.annotated_program_plan",
    )
    plan_path = directory / string(
        artifact.get("path"), "selection.annotated_program_plan.path"
    )
    payload = plan_path.read_text()
    expected_artifact = string(
        artifact.get("artifact_sha256"),
        "selection.annotated_program_plan.artifact_sha256",
    )
    if text_digest(payload) != expected_artifact:
        raise ValueError("saved annotated-plan artifact digest does not match")
    plan = AnnotatedProgramPlan.from_json(payload)
    expected_plan = string(
        artifact.get("plan_digest"), "selection.annotated_program_plan.plan_digest"
    )
    if plan.digest != expected_plan:
        raise ValueError("saved annotated-plan content digest does not match")
    if manifest.get("memory_budgets") != plan.memory_budgets.to_dict():
        raise ValueError("selection manifest memory budgets do not match the plan")
    if manifest.get("transfer_bandwidths") != plan.transfer_bandwidths.to_dict():
        raise ValueError("selection manifest transfer bandwidths do not match the plan")
    return plan


__all__ = ["load_annotated_plan", "save_annotated_plan"]
