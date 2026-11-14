"""Verify recorded decisions and pack arrays separately from shared labels."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from shadowspill.diagnostics.pressurefit_trace import PressureFitTrace, read_manifest

from .io import write


def export(args):
    directory = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    collection = directory / "collection.json"
    if collection.is_file():
        records = []
        for path in json.loads(collection.read_text())["directories"]:
            trace_directory = Path(path)
            destination = output / trace_directory.parent.name
            destination.mkdir()
            records.append(export_directory(trace_directory, destination))
        write(
            output / "collection.json", {"source": str(directory), "programs": records}
        )
    else:
        export_directory(directory, output)


def export_directory(directory, output):
    manifest = read_manifest(directory)
    summary: dict[str, Any] = {
        "source": str(directory),
        "candidates": [],
        "reductions": 0,
        "cuts": 0,
        "bytes": 0,
    }
    semantics = directory / "semantic_labels.json"
    if semantics.is_file():
        write(output / "semantic_labels.json", json.loads(semantics.read_text()))
        summary["semantic_labels"] = "semantic_labels.json"
    # Shared resolution metadata is labelled once. Candidate JSON references it;
    # do not expand the same large alias table for every policy in the export.
    for index, context in enumerate(manifest.get("contexts", ())):
        write(output / f"context_{index:04d}.json", context)
    for index, candidate in enumerate(manifest["candidates"]):
        trace = PressureFitTrace.read(directory / candidate["file"])
        started = time.perf_counter()
        trace.verify()
        arrays = {
            name: getattr(trace, name)
            for name in (
                "header",
                "alias_sizes",
                "seed",
                "initial",
                "step_reductions",
                "steps",
                "outcome",
            )
        }
        for reduction, value in enumerate(trace.reductions):
            for name in ("changes", "cuts", "pressure_ops", "outcome"):
                arrays[f"reduction_{reduction:04d}_{name}"] = getattr(value, name)
            if value.reason is not None:
                arrays[f"reduction_{reduction:04d}_reason"] = value.reason
        np.savez_compressed(output / f"candidate_{index:04d}.npz", **arrays)
        candidate.update(
            reductions=len(trace.reductions),
            cuts=sum(len(r.cuts) for r in trace.reductions),
            answer_step=trace.answer_step,
            answer_reduction=trace.answer_reduction,
            reduction_reasons=[
                trace.reduction_reason(rid) for rid in range(len(trace.reductions))
            ],
            bytes=(directory / candidate["file"]).stat().st_size,
            verification_seconds=time.perf_counter() - started,
        )
        metadata = dict(candidate)
        if "context_index" in metadata:
            metadata.pop("aliases")
            metadata.pop("task_ids")
            metadata["context_file"] = f"context_{metadata['context_index']:04d}.json"
        # Boundary labels are derived from tasks and sidecar metadata by the reader.
        metadata.pop("boundaries")
        write(output / f"candidate_{index:04d}.json", metadata)
        summary["candidates"].append(
            {
                key: candidate[key]
                for key in (
                    "candidate_id",
                    "context_index",
                    "reductions",
                    "cuts",
                    "bytes",
                    "verification_seconds",
                )
                if key in candidate
            }
            | {
                "metadata": f"candidate_{index:04d}.json",
                "arrays": f"candidate_{index:04d}.npz",
            }
        )
        for key in ("reductions", "cuts", "bytes"):
            summary[key] += candidate[key]
        print(
            "EXPORT",
            {
                key: candidate[key]
                for key in (
                    "candidate_id",
                    "reductions",
                    "cuts",
                    "bytes",
                    "verification_seconds",
                )
            },
            flush=True,
        )
    write(output / "manifest.json", summary)
    return summary
