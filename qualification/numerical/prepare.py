"""Construct and export the numerical case, before measurement/execution workers."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from benchmarking.spill import record, resolve

from .metrics import state_digest
from .references import reference_artifact_exists, reference_inputs_path
from .request import CaseRequest, PlannedRequest, workload_metadata_for
from .tolerances import SPILL_BUDGET


def save_request(request: PlannedRequest, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    value = {
        **asdict(replace(request, spill_pool=None)),
        "spill_pool": record(request.spill_pool),
    }
    (directory / "request.json").write_text(
        json.dumps(value, default=str, indent=2) + "\n"
    )
    settings = {
        "execution_bytes": request.device_budget,
        "external_headroom_bytes": request.external_headroom_mib << 20,
        "spill_bytes": (
            SPILL_BUDGET if request.spill_pool is None else request.spill_pool.capacity
        ),
        "spill": record(request.spill_pool),
        "optimizer_ordering": request.case.optimizer_ordering,
        "data_ordering": request.case.data_ordering,
        "deterministic": True,
    }
    (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


def load_request(directory: Path) -> PlannedRequest:
    value = json.loads((directory / "request.json").read_text())
    value["case"] = CaseRequest(**value["case"])
    value["spill_pool"] = resolve({"spill": value["spill_pool"]})
    for name in ("reference_path", "result_path"):
        if value.get(name) is not None:
            value[name] = Path(value[name])
    return PlannedRequest(**value)


def run(directory: Path) -> None:
    from qualification.pipeline.build import build_case

    request = load_request(directory)
    if not reference_artifact_exists(request.reference_path):
        raise RuntimeError("compiled reference is incomplete; regenerate it first")
    case = request.case.build()
    reference = torch.load(
        reference_inputs_path(request.reference_path),
        map_location="cpu",
        weights_only=True,
    )
    digest = state_digest(case.microbatches)
    if state_digest(reference) != digest:
        raise RuntimeError("reference inputs differ; use --regenerate-reference")
    (directory / "case.json").write_text(
        json.dumps(
            {
                "input_digest": digest,
                "workload_metadata": workload_metadata_for(
                    case, request.profiling_metadata
                ),
            },
            indent=2,
        )
        + "\n"
    )
    with case.implementations(deterministic=True):
        build_case(case, directory, dtypes=request.case.dtypes, deterministic=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
