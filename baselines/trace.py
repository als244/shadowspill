"""Trace one case of an experiment under Nsight Systems, with the GPU's
metrics sampled.

    python -m baselines.trace DIR SWEEP CASE [--warmup-steps N] [--steps N] \\
        [--devices LIST] [--capture measured|whole] [--trace APIS] \\
        [--gpu-metrics-devices DEVICES] [--gpu-metrics-frequency HZ] \\
        [--gpu-metrics-set NAME] [--[no-]cuda-memory-usage] \\
        [--cuda-graph-trace graph|node] [--export FORMATS] \\
        [--nsys-arg ARG ...] [--output DIR] [--name NAME] [--nsys PATH] \\
        [--timeout SECONDS] [--dry-run]

DIR is an experiment's output directory (``baselines.experiment``), SWEEP one
of its sweeps and CASE one of that sweep's cases, by its name or the digest
that ends it. The case runs again as its sweep ran it, with the sweep's
engine, world size and trainer interpreter and the case's settings, with
any saved preparation from the case's launch command, and ``nsys_capture``
on, so the runner opens the capture at the first measured
update and closes it after the last; ``--warmup-steps`` and ``--steps``
change how many updates run, ``--devices`` the GPUs it runs on
(``CUDA_VISIBLE_DEVICES``), and ``--capture whole`` records the whole run,
setup and planning included, instead of the measured updates. ``nsys
profile`` records the calls of ``--trace`` (CUDA, NVTX, cuDNN, cuBLAS and
the OS runtime), the device memory in use, CUDA graphs whole or node by
node, and the metrics of the GPUs ``--gpu-metrics-devices`` names (indices,
``all`` or ``cuda-visible``) at ``--gpu-metrics-frequency``, without CPU
sampling; ``--export`` writes the report again as SQLite or another format,
and ``--nsys-arg`` passes any other option of ``nsys profile`` through.
Everything is written under ``--output``, by default
``DIR/traces/<sweep>/<case>/<stamp>/``: ``<name>.nsys-rep``, the command,
the console of nsys and the trainer, ``trace.json`` with what was asked,
and the trainer's own files with the row it wrote, which says it is not a
throughput result. The sweeps, the summary and the report never read a
trace's directory.

FSDP2's eager checkpoints and DeepSpeed's checkpoints also label the original
forward and the actual recomputation with ``baseline.phase`` NVTX ranges.
FSDP2 labels the full forward/backward calls and both engines label optimizer
calls. These are host launch scopes: correlate CUDA APIs with kernel events
to measure GPU work instead of using the host range durations. Compiled FSDP2
blocks retain their compiler checkpoint context and do not get replay scopes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import shutil
from pathlib import Path

from baselines.common import launch, write_json
from baselines.sweep import clock, ended_block, run_case


def find_case(sweep: Path, case: str) -> Path:
    """The case directory a name or a digest names, exactly one."""

    matches = [
        path
        for path in sorted((sweep / "cases").glob("*__*"))
        if path.name == case or path.name.endswith(f"__{case}")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{case!r} names {len(matches)} cases of {sweep.name}; give a whole name"
        )
    return matches[0]


def profile_command(nsys: str, arguments, output: Path, command: list[str]):
    """``nsys profile`` around one case's launch command."""

    options = [
        f"--output={output / arguments.name}",
        "--force-overwrite=true",
        f"--trace={arguments.trace}",
        f"--cuda-memory-usage={str(arguments.cuda_memory_usage).lower()}",
        f"--cuda-graph-trace={arguments.cuda_graph_trace}",
        f"--gpu-metrics-devices={arguments.gpu_metrics_devices}",
        f"--gpu-metrics-frequency={arguments.gpu_metrics_frequency}",
        "--sample=none",
        "--cpuctxsw=none",
    ]
    if arguments.capture == "measured":
        options += ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
    if arguments.gpu_metrics_set:
        options.append(f"--gpu-metrics-set={arguments.gpu_metrics_set}")
    if arguments.export:
        options.append(f"--export={arguments.export}")
    return [nsys, "profile", *options, *arguments.nsys_arg, *command]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="an experiment's output")
    parser.add_argument("sweep", help="the sweep the case belongs to")
    parser.add_argument("case", help="the case's name, or the digest that ends it")
    parser.add_argument("--warmup-steps", type=int, help="default: the case's")
    parser.add_argument(
        "--steps", type=int, help="measured updates; default: the case's"
    )
    parser.add_argument(
        "--devices",
        help="the GPUs the case runs on, as CUDA_VISIBLE_DEVICES lists them",
    )
    parser.add_argument(
        "--capture",
        choices=("measured", "whole"),
        default="measured",
        help="the measured updates, or the whole run with setup and planning",
    )
    parser.add_argument(
        "--trace",
        default="cuda,nvtx,osrt,cudnn,cublas",
        help="the APIs nsys traces",
    )
    parser.add_argument(
        "--gpu-metrics-devices",
        default="all",
        help="GPU indices, 'all' or 'cuda-visible' (nsys's names)",
    )
    parser.add_argument(
        "--gpu-metrics-frequency", type=int, default=10000, help="samples per second"
    )
    parser.add_argument("--gpu-metrics-set", help="default: nsys's for the device")
    parser.add_argument(
        "--cuda-memory-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="track the device memory in use",
    )
    parser.add_argument(
        "--cuda-graph-trace",
        choices=("graph", "node"),
        default="graph",
        help="CUDA graphs whole, or node by node at a higher overhead",
    )
    parser.add_argument(
        "--export", help="further formats of the report, such as sqlite"
    )
    parser.add_argument(
        "--nsys-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="any other option of nsys profile, as --nsys-arg=--delay=5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="the trace's directory (default: DIR/traces/SWEEP/CASE/<stamp>)",
    )
    parser.add_argument(
        "--name", default="trace", help="the report's file name, without .nsys-rep"
    )
    parser.add_argument("--nsys", help="the nsys executable (default: on PATH)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200,
        help="seconds for the case and nsys's own processing",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="write the command; launch nothing"
    )
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    sweep = directory / "sweeps" / args.sweep
    if not (sweep / "manifest.json").is_file():
        parser.error(f"no sweep {args.sweep!r} under {args.directory}")
    nsys = args.nsys or shutil.which("nsys")
    if nsys is None:
        parser.error("nsys is not on PATH; name it with --nsys")
    if not args.name or "/" in args.name:
        parser.error("the report's name is a plain file name")
    try:
        case = find_case(sweep, args.case)
        manifest = json.loads((sweep / "manifest.json").read_text())
        module = launch.engine_module(manifest["engine"])
        values = json.loads((case / "config.json").read_text())
        values["nsys_capture"] = True
        for name in ("warmup_steps", "steps"):
            if getattr(args, name) is not None:
                values[name] = getattr(args, name)
        settings = module.Settings.from_dict(values)
        settings.validate(manifest["world_size"])
    except ValueError as error:
        parser.error(str(error))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.output or directory / "traces" / args.sweep / case.name / stamp
    ).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    trainer = launch.command(
        manifest["engine"],
        manifest["python"],
        manifest["world_size"],
        settings,
        output,
    )
    recorded_command = case / "command.txt"
    if recorded_command.is_file():
        recorded = shlex.split(recorded_command.read_text())
        if "--prepared" in recorded:
            index = recorded.index("--prepared")
            trainer += ["--prepared", recorded[index + 1]]
    command = profile_command(nsys, args, output, trainer)
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    write_json(
        output / "trace.json",
        {
            "experiment": str(directory),
            "sweep": args.sweep,
            "case": case.name,
            "devices": args.devices,
            "capture": args.capture,
            "settings": settings.as_dict(),
            "command": command,
        },
    )
    print(f"TRACE {args.sweep}/{case.name}", flush=True)
    print(f"  output: {output}")
    print(f"  command: {output / 'command.txt'}", flush=True)
    if args.dry_run:
        print("  dry run: command written", flush=True)
        return
    if args.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    print(f"  start: {clock()}; console: {output / 'console.log'}", flush=True)
    outcome = run_case(command, output, settings, manifest["world_size"], args.timeout)
    print("\n".join(ended_block("", args.case, outcome, output)), flush=True)
    report = output / f"{args.name}.nsys-rep"
    if report.is_file():
        print(f"  trace: {report} ({report.stat().st_size / 2**20:,.1f} MiB)")
    else:
        print(f"  no trace written; see {output / 'console.log'}")


if __name__ == "__main__":
    main()
