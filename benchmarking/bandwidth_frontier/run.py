"""Load profiles once; persist each calibration so an interrupted sweep resumes."""

import hashlib
import json
from dataclasses import replace

from shadowspill import CommunicationInfo, MemoryBudgets, ProfiledProgram, SweepResult

from .planning import plan
from .results import collect_measured, rows, write_tables

GIB = 1 << 30


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args, paths):
    programs = {name: ProfiledProgram.load(path) for name, path in paths.items()}
    baseline = (
        CommunicationInfo.load(args.communication_info)
        if args.communication_info
        else CommunicationInfo(1, 1)
    )
    latencies = args.latency_us or [
        baseline.fetch_latency_seconds * 1e6,
        baseline.evict_latency_seconds * 1e6,
    ]
    request = {
        "programs": {name: file_digest(path) for name, path in paths.items()},
        "budgets_gib": args.budget_gib,
        "spill_gib": args.spill_gib,
        "calibrations": args.sim_transfer_bandwidths,
        "communication": baseline.to_dict(),
        "latency_us": latencies,
        "orderings": args.orderings,
        "resolution_options": args.resolution_options,
        "search_workers": args.search_workers,
        "units_per_step": args.units_per_step,
        "unit_label": args.unit_label,
        "title": args.title or ", ".join(programs),
        "measured_digest": None
        if args.measured is None
        else file_digest(args.measured),
    }
    request = json.loads(json.dumps(request))
    root = args.output_dir / "raw_data"
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "sweep.json"
    if request_path.exists():
        if not args.resume or json.loads(request_path.read_text()) != request:
            raise ValueError(
                "use a fresh output directory or --resume with unchanged inputs"
            )
    elif args.resume:
        raise ValueError("--resume requires an existing raw_data/sweep.json")
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    budgets = [
        MemoryBudgets(int(device * GIB), int(spill * GIB))
        for spill in args.spill_gib
        for device in args.budget_gib
    ]
    records = []
    for index, (fetch, evict) in enumerate(args.sim_transfer_bandwidths):
        path = root / f"calibration-{index:04d}.ssp"
        if args.resume and path.exists():
            result = SweepResult.load(path)
            print(f"{fetch:g}/{evict:g} GB/s: loaded {path}", flush=True)
        else:
            communication = replace(
                baseline,
                fetch_bytes_per_second=fetch * 1e9,
                evict_bytes_per_second=evict * 1e9,
                fetch_concurrent_bytes_per_second=fetch * 1e9,
                evict_concurrent_bytes_per_second=evict * 1e9,
                fetch_latency_seconds=latencies[0] / 1e6,
                evict_latency_seconds=latencies[1] / 1e6,
            )
            print(f"{fetch:g}/{evict:g} GB/s: planning", flush=True)
            result = plan(programs, budgets, communication, args)
            result.save(path)
            print(f"{fetch:g}/{evict:g} GB/s: saved {path}", flush=True)
        records.extend(rows(result, fetch, evict, args.units_per_step))
    write_tables(root, records)
    if args.measured is not None:
        collect_measured(args.measured, root)
    if args.plots:
        from .plot import draw

        draw(root, args.output_dir)
