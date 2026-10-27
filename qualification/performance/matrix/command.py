"""Construct one performance cell command from its resolved manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from qualification.performance.cases import FullModelManifest, manifest_for
from qualification.pipeline.process import python_command
from qualification.precision import dtype_arguments


def cell_command(
    manifest: FullModelManifest,
    arguments: argparse.Namespace,
    artifact: Path,
    planning_budgets: dict[str, int],
) -> list[str]:
    """Launch one cell; its phase archives sit beside its result artifact."""

    command = python_command(
        "qualification.performance.run",
        manifest.family,
        manifest.implementation,
        artifact,
    )
    if arguments.plan_only:
        command.append("--plan-only")
    elif not arguments.checkpoint:
        # Checkpoint replay is covered by the numerical gate; opt in here.
        command.append("--skip-checkpoint")
    if arguments.measure_only:
        command.append("--measure-only")
    if manifest.identity in planning_budgets:
        command.extend(
            ("--planning-spill-budget-gib", str(planning_budgets[manifest.identity]))
        )
    command.extend(
        (
            "--execution-budget-gib",
            str(manifest.device_physical_capacity_bytes >> 30),
        )
    )
    command.extend(dtype_arguments(manifest))
    command.extend(
        ("--external-headroom-mib", str(manifest.external_headroom_bytes >> 20))
    )
    canonical = manifest_for(manifest.family, manifest.implementation)
    if manifest.spill_budget_bytes != canonical.spill_budget_bytes:
        command.extend(("--spill-budget-gib", str(manifest.spill_budget_bytes >> 30)))
    remote_spill = getattr(arguments, "remote_spill", None)
    if remote_spill is not None:
        command.extend(("--remote-spill", remote_spill))
    return command
