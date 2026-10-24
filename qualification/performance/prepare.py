"""Build a performance cell and save everything its later phases need."""

import json
import sys
from argparse import Namespace
from dataclasses import fields
from pathlib import Path

from qualification.precision import dtype_overrides

from .cases import FullModelManifest
from .manifest import _manifest_with_overrides


def manifest_for_run(arguments: Namespace) -> FullModelManifest:
    return _manifest_with_overrides(
        arguments.family,
        arguments.implementation,
        spill_budget_gib=arguments.spill_budget_gib,
        execution_budget_gib=arguments.execution_budget_gib,
        external_headroom_mib=arguments.external_headroom_mib,
        reject_overbudget=True,
        **dtype_overrides(arguments),
    )


def read_request(directory: Path) -> tuple[Namespace, FullModelManifest]:
    from workloads.models import catalog

    arguments = Namespace(**json.loads((directory / "arguments.json").read_text()))
    arguments.output = Path(arguments.output)
    value = json.loads((directory / "manifest.json").read_text())
    value["model_config"] = catalog.entry(value["model"]).config(
        **value["model_config"]
    )
    return arguments, FullModelManifest(
        **{f.name: value[f.name] for f in fields(FullModelManifest)}
    )


def run(directory: Path) -> None:
    from qualification.pipeline.build import build_case as build_program
    from workloads.full_model import build_case

    arguments, manifest = read_request(directory)
    case = build_case(manifest, seed=arguments.seed)
    with case.implementations():
        build_program(case, directory, dtypes=manifest.dtypes)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
