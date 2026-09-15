"""The two fresh processes one case needs: the reference, then the plan.

Each arm runs in its own interpreter because a compiled reference and a
planned run must not share a process: the allocator, the caches and the
compiler state of one would otherwise be visible to the other.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from workloads.numerical import ModelImplementation

from .references import canonical_reference_path, reference_artifact_exists


def orchestrate(
    family: str,
    model_implementation: ModelImplementation,
    result_directory: Path,
    device_budget: int,
    *,
    seed: int,
    model_config_argument: str,
    data_geometry_argument: str | None,
    case_factory: str | None,
    case_option_arguments: list[str],
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    data_ordering: str | None,
    steps: int,
    checkpoint_step: int,
    require_pressure: bool,
    artifact_store: Path | None,
    build_store: Path | None,
    plan_store: Path | None,
    profiling_metadata_argument: str | None,
    build_store_mode: str,
    plan_store_mode: str,
    export_bypass_key: str | None,
    reference_directory: Path,
    regenerate_reference: bool,
    detailed_artifacts: bool,
) -> None:
    result_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_implementation}_{family}"
    reference = canonical_reference_path(
        reference_directory,
        model_name=family,
        implementation=model_implementation,
    )
    result = result_directory / f"{prefix}.json"
    base = [sys.executable, "-m", "tools.qualification.numerical"]
    options = [
        "--seed",
        str(seed),
        "--model-config",
        model_config_argument,
        "--optimizer-ordering",
        optimizer_ordering,
    ]
    options.extend(("--steps", str(steps)))
    if not require_pressure:
        options.append("--allow-fully-resident")
    if data_geometry_argument is not None:
        options.extend(("--data-geometry", data_geometry_argument))
    if data_ordering is not None:
        options.extend(("--data-ordering", data_ordering))
    if profiling_metadata_argument is not None:
        options.extend(("--profiling-metadata", profiling_metadata_argument))
    if case_factory is not None:
        options.extend(("--case-factory", case_factory))
    for value in case_option_arguments:
        options.extend(("--case-option", value))
    environment = dict(os.environ)
    if regenerate_reference or not reference_artifact_exists(reference):
        subprocess.run(
            [
                *base,
                "_reference",
                family,
                str(reference),
                "--model-implementation",
                model_implementation,
                *options,
            ],
            check=True,
            env=environment,
        )
    planned_options: list[str] = []
    selected_cache = artifact_store or result_directory / "artifact_store"
    planned_options.extend(("--artifact-store", str(selected_cache)))
    for flag, path in (("--build-store", build_store), ("--plan-store", plan_store)):
        if path is not None:
            planned_options.extend((flag, str(path)))
    for tree, mode in (("build", build_store_mode), ("plan", plan_store_mode)):
        if mode != "contribute":
            planned_options.extend((f"--{tree}-store-mode", mode))
    if export_bypass_key is not None:
        planned_options.extend(("--export-bypass-key", export_bypass_key))
    if detailed_artifacts:
        planned_options.append("--detailed-artifacts")
    subprocess.run(
        [
            *base,
            "_planned",
            family,
            str(reference),
            str(result),
            str(device_budget),
            "--model-implementation",
            model_implementation,
            *options,
            *planned_options,
            "--checkpoint-step",
            str(checkpoint_step),
        ],
        check=True,
        env=environment,
    )
    print(result.read_text(), end="")
