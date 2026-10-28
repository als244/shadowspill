"""Describe one numerical case and construct its isolated worker commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qualification.numerical.references import reference_artifact_exists
from qualification.numerical.spill import spill_arguments
from qualification.pipeline.process import python_command
from qualification.precision import dtype_arguments
from shadowspill.memory import SpillPool


@dataclass(frozen=True, slots=True)
class CaseOptions:
    """What every case in the matrix is run the same way with."""

    environment: dict[str, str]
    reference_directory: Path
    regenerate_reference: bool
    seed: int
    model_config: str
    data_geometry: str | None
    case_factory: str | None
    case_options: list[str]
    optimizer_ordering: str
    data_ordering: str | None
    empty_caches: bool
    cache_directory: Path | None
    detailed_artifacts: bool
    spill_pool: SpillPool | None = None
    external_headroom_mib: int | None = None
    reject_overbudget: bool = True
    model_dtype: str | None = None
    master_dtype: str | None = None
    grad_dtype: str | None = None
    opt_state_dtype: str | None = None
    retain_execution_files: bool = False

    def case_arguments(self) -> list[str]:
        """The options both arms of a case are given, in their fixed order."""

        arguments = [
            "--seed",
            str(self.seed),
            "--model-config",
            self.model_config,
            "--optimizer-ordering",
            self.optimizer_ordering,
        ]
        arguments.extend(dtype_arguments(self))
        if self.data_geometry is not None:
            arguments.extend(("--data-geometry", self.data_geometry))
        if self.data_ordering is not None:
            arguments.extend(("--data-ordering", self.data_ordering))
        if self.case_factory is not None:
            arguments.extend(("--case-factory", self.case_factory))
        for value in self.case_options:
            arguments.extend(("--case-option", value))
        # Only the planned arm spills. The reference arm is the unplanned run
        # the planned one is compared against, so giving it a remote pool would
        # change the thing being measured against.
        return arguments


def case_commands(
    family: str,
    implementation: str,
    device_budget: int,
    reference: Path,
    artifact: Path,
    options: CaseOptions,
) -> list[list[str]]:
    """The warm-up, the reference arm when one must be made, and the planned arm.

    The warm-up process compiles and runs every microbatch once and exits, so
    kernels that tune on first use (Quack's autotuner under the SonicMoE
    implementations) are tuned before either arm starts: a process that tunes
    a kernel computes slightly differently from one that reads the tuned
    choice, for the rest of its life, and both arms must be readers.
    """

    base = python_command("qualification.numerical.run")
    shared = options.case_arguments()
    commands: list[list[str]] = [
        [*base, "_warm", family, "--model-implementation", implementation, *shared]
    ]
    if options.regenerate_reference or not reference_artifact_exists(reference):
        commands.append(
            [
                *base,
                "_reference",
                family,
                str(reference),
                "--model-implementation",
                implementation,
                *shared,
            ]
        )
    planned = [
        *base,
        "_planned",
        family,
        str(reference),
        str(artifact),
        str(device_budget),
        "--model-implementation",
        implementation,
        *shared,
    ]
    if options.external_headroom_mib is not None:
        planned.extend(("--external-headroom-mib", str(options.external_headroom_mib)))
    planned.extend(spill_arguments(options.spill_pool))
    if options.retain_execution_files:
        planned.append("--retain-execution-files")
    commands.append(planned)
    return commands
