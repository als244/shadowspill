"""Canonical compiled-reference storage for numerical qualification."""

from __future__ import annotations

from pathlib import Path

from shadowspill.schema import artifact_schema

REFERENCE_SCHEMA = artifact_schema("compiled_reference")
DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY = Path(
    "qualification/results/references/approximately_1b"
)


def canonical_reference_path(
    root: Path,
    *,
    model_name: str,
    implementation: str,
) -> Path:
    """Return the single identity-checked reference slot for one provider cell."""

    return root.expanduser().resolve() / model_name / implementation / "reference.pt"


def reference_inputs_path(reference: Path) -> Path:
    """Return the exact-input sidecar belonging to one reference state."""

    return reference.with_name("inputs.pt")


def reference_artifact_exists(reference: Path) -> bool:
    """Return whether both required files of a compact reference exist."""

    return reference.is_file() and reference_inputs_path(reference).is_file()


__all__ = [
    "DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY",
    "REFERENCE_SCHEMA",
    "canonical_reference_path",
    "reference_artifact_exists",
    "reference_inputs_path",
]
