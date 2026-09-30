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
    execution_budget_gib: int | None = None,
    external_headroom_mib: int | None = None,
    reject_overbudget: bool = False,
    model_dtype: str | None = None,
    master_dtype: str | None = None,
    grad_dtype: str | None = None,
    opt_state_dtype: str | None = None,
) -> FullModelManifest:
    """Resolve one immutable qualification manifest and CLI overrides."""

    manifest = manifest_for(family, cast(ModelImplementation, implementation))
    if spill_budget_gib is not None:
        if spill_budget_gib <= 0:
            raise ValueError("spill-budget-gib must be positive")
        manifest = replace(manifest, spill_budget_bytes=spill_budget_gib << 30)
    if execution_budget_gib is not None:
        if execution_budget_gib <= 0:
            raise ValueError("execution-budget-gib must be positive")
        manifest = replace(
            manifest, device_physical_capacity_bytes=execution_budget_gib << 30
        )
    if external_headroom_mib is not None:
        headroom = external_headroom_mib << 20
        if not 0 <= headroom < manifest.device_physical_capacity_bytes:
            raise ValueError(
                "external-headroom-mib must be nonnegative and below the execution cap"
            )
        manifest = replace(manifest, external_headroom_bytes=headroom)
    overrides = {
        key: value
        for key, value in (
            ("model_dtype", model_dtype),
            ("master_dtype", master_dtype),
            ("grad_dtype", grad_dtype),
            ("opt_state_dtype", opt_state_dtype),
        )
        if value is not None
    }
    manifest = replace(manifest, reject_overbudget=reject_overbudget, **overrides)
    _ = manifest.dtypes  # Validate before creating a runtime or allocating pools.
    return manifest


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
