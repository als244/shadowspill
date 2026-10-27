"""Run or repeat one phase without importing the other phases."""

import argparse
import json
from pathlib import Path

from .orchestrate import run, run_request
from .phases.common import read
from .spill import add_arguments, from_arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "build", "profile", "communicate", "plan", "execute"),
        default="all",
    )
    parser.add_argument("--factory")
    parser.add_argument("--factory-args", type=json.loads)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-budget-gib", type=float, nargs="+")
    parser.add_argument("--run-budget-gib", type=float, nargs="*")
    parser.add_argument("--spill-gib", type=float)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--communication-info", type=Path)
    parser.add_argument("--rank-map", type=json.loads)
    parser.add_argument("--warmup-seconds", type=float)
    parser.add_argument("--measure-seconds", type=float)
    parser.add_argument("--external-headroom-gib", type=float)
    parser.add_argument("--search-workers", type=int)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.phase == "all" or (
        args.phase == "build" and not (args.output_dir / "request.json").exists()
    ):
        if args.factory is None or (
            args.phase == "all"
            and (args.search_budget_gib is None or args.spill_gib is None)
        ):
            parser.error("all requires --factory, --search-budget-gib and --spill-gib")
        storage_flags = {
            "ssd_spill",
            "remote_spill",
            "ssd_staging_mib",
            "ssd_chunk_mib",
            "ssd_queue_depth",
        }
        values = {
            k: v
            for k, v in vars(args).items()
            if v is not None and k not in {"phase", *storage_flags}
        }
        try:
            values["spill"] = from_arguments(
                args, int((args.spill_gib or 0) * (1 << 30))
            )
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        if args.communication_info is not None:
            from shadowspill import CommunicationInfo

            values["communication_info"] = CommunicationInfo.load(
                args.communication_info
            )
        return run(**values, phase=args.phase)
    request_file = args.output_dir / "request.json"
    if not request_file.is_file():
        parser.error("a named phase needs an existing output-dir/request.json")
    request = read(request_file)
    for name in (
        "factory",
        "factory_args",
        "device",
        "steps",
        "warmup_seconds",
        "measure_seconds",
        "search_workers",
        "plots",
        "rank_map",
    ):
        value = getattr(args, name)
        if value is not None:
            request[name] = value
    for name, key in (
        ("search_budget_gib", "search_budgets"),
        ("run_budget_gib", "run_budgets"),
    ):
        value = getattr(args, name)
        if value is not None:
            request[key] = [int(v * (1 << 30)) for v in value]
    if args.search_budget_gib is not None and args.run_budget_gib is None:
        request["run_budgets"] = [
            b for b in request["run_budgets"] if b in request["search_budgets"]
        ]
    for name, key in (
        ("spill_gib", "spill_bytes"),
        ("external_headroom_gib", "external_headroom_bytes"),
    ):
        value = getattr(args, name)
        if value is not None:
            request[key] = int(value * (1 << 30))
    request["output_dir"] = str(args.output_dir.resolve())
    try:
        selected = from_arguments(args, request["spill_bytes"])
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if selected is not None:
        request["spill"] = selected
    elif request.get("spill"):
        request["spill"]["arguments"]["capacity"] = request["spill_bytes"]
    if args.communication_info is not None:
        request["communication_info"] = str(args.communication_info.resolve())
    return run_request(request, args.phase)
