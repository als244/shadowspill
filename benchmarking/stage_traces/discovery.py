"""Capture inventories in fresh processes, then enumerate unique trace cases."""

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

from benchmarking._serialization import atomic_json
from benchmarking.processes import worker_command

from .process import launch


def prepare_run(args):
    args.output = args.output.absolute()
    for child in ("scripts", "logs", "notes", "results", "graphs"):
        (args.output / child).mkdir(parents=True, exist_ok=True)
    options = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output", "resume", "retry_failed", "discover_only"}
    }
    package = Path(__file__).parent
    sources = [
        *package.glob("*.py"),
        package.parent / "processes.py",
        package.parent / "_serialization.py",
    ]
    manifest = {
        "options": options,
        "source_hashes": {
            str(path.relative_to(package.parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(sources)
        },
        "python": sys.executable,
        "no_site": bool(sys.flags.no_site),
    }
    path = args.output / "run-manifest.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != manifest:
            raise ValueError("resume requires the same tracing source/settings")
    else:
        atomic_json(path, manifest)
    return options


def discover(args, gpu):
    manifests = []
    for length, batch in itertools.product(args.sequence_lengths, args.batch_sizes):
        directory = args.output / "graphs" / f"seqlen-{length}" / f"batch-{batch}"
        if not (directory / "manifest.json").exists():
            # Retain failed discovery output; only a complete inventory is reusable.
            if directory.exists():
                directory.rename(
                    directory.with_name(directory.name + f".failed-{time.time_ns()}")
                )
            command = worker_command(
                "benchmarking.stage_traces.capture",
                [
                    "--factory",
                    args.factory,
                    "--factory-args",
                    json.dumps(args.factory_args),
                    "--sequence-length",
                    length,
                    "--batch-size",
                    batch,
                    "--output",
                    directory,
                    "--variants",
                    *args.variants,
                ],
            )
            print(f"DISCOVER seqlen={length} batch={batch}", flush=True)
            code, timed_out = launch(
                command,
                args.output / "logs" / f"discover-s{length}-b{batch}.log",
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                timeout=args.discovery_timeout,
            )
            if code or timed_out:
                raise RuntimeError(
                    f"discovery failed for {length}/{batch}; inspect logs"
                )
        manifests.append(json.loads((directory / "manifest.json").read_text()))
    return describe_queue(manifests, args.output)


def describe_queue(manifests, output):
    cases = []
    for manifest in manifests:
        for stage in manifest["stages"]:
            for variant in stage["variants"]:
                case = {
                    "stage": stage["name"],
                    "sequence_length": manifest["sequence_length"],
                    "batch_size": manifest["batch_size"],
                    "variant": variant["name"],
                    "artifact": variant["artifact"],
                    "occurrences": stage["occurrences"],
                }
                case["directory"] = str(
                    output
                    / "results"
                    / stage["name"]
                    / f"seqlen-{case['sequence_length']}"
                    / f"batch-{case['batch_size']}"
                    / case["variant"]
                )
                cases.append(case)
    return cases
