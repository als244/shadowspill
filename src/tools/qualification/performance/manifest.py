"""The cell's manifest, and the budgets an override may narrow."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from workloads.full_model import FullModelManifest, manifest_for
from workloads.providers import ModelImplementation


def _manifest_with_overrides(
    family: str,
    implementation: str,
    *,
    spill_budget_gib: int | None,
) -> FullModelManifest:
    """Resolve one immutable qualification manifest and CLI overrides."""

    manifest = manifest_for(family, cast(ModelImplementation, implementation))
    if spill_budget_gib is None:
        return manifest
    if spill_budget_gib <= 0:
        raise ValueError("spill-budget-gib must be positive")
    return replace(manifest, spill_budget_bytes=spill_budget_gib << 30)


def _planning_spill_budget(
    manifest: FullModelManifest,
    *,
    planning_spill_budget_gib: int | None,
) -> int:
    """Resolve a plan budget bounded by the runtime spill-pool capacity."""

    if planning_spill_budget_gib is None:
        return manifest.spill_budget_bytes
    if planning_spill_budget_gib <= 0:
        raise ValueError("planning-spill-budget-gib must be positive")
    budget = planning_spill_budget_gib << 30
    if budget > manifest.spill_budget_bytes:
        raise ValueError(
            "planning spill budget exceeds the configured runtime spill pool: "
            f"budget={budget}, capacity={manifest.spill_budget_bytes}"
        )
    return budget
