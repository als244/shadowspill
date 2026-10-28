"""The two fresh processes one case needs: the reference, then the plan.

Each arm runs in its own interpreter because a compiled reference and a
planned run must not share a process: the allocator, the caches and the
compiler state of one would otherwise be visible to the other.
"""

from __future__ import annotations

import os
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Literal

from qualification.pipeline.process import python_command
from qualification.precision import dtype_arguments
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
    profiling_metadata_argument: str | None,
    reference_directory: Path,
    regenerate_reference: bool,
    detailed_artifacts: bool,
    external_headroom_mib: int | None = None,
    reject_overbudget: bool = True,
    model_dtype: str | None = None,
    master_dtype: str | None = None,
    grad_dtype: str | None = None,
    opt_state_dtype: str | None = None,
    spill_options: list[str] | None = None,
    retain_execution_files: bool = False,
) -> None:
    result_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_implementation}_{family}"
    reference = canonical_reference_path(
        reference_directory,
        model_name=family,
        implementation=model_implementation,
    )
    result = result_directory / f"{prefix}.json"
    base = python_command("qualification.numerical.run")
    options = [
        "--seed",
        str(seed),
        "--model-config",
        model_config_argument,
        "--optimizer-ordering",
        optimizer_ordering,
    ]
    options.extend(
        dtype_arguments(
            Namespace(
                model_dtype=model_dtype,
                master_dtype=master_dtype,
                grad_dtype=grad_dtype,
                opt_state_dtype=opt_state_dtype,
            )
        )
    )
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
    # The warm-up process tunes and compiles first, so neither arm is the
    # process that tuned (matrix.commands.case_commands says why).
    subprocess.run(
        [
            *base,
            "_warm",
            family,
            "--model-implementation",
            model_implementation,
            *options,
        ],
        check=True,
        env=environment,
    )
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
    planned_options = list(spill_options or ())
    if external_headroom_mib is not None:
        planned_options.extend(("--external-headroom-mib", str(external_headroom_mib)))
    if detailed_artifacts:
        planned_options.append("--detailed-artifacts")
    if retain_execution_files:
        planned_options.append("--retain-execution-files")
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
