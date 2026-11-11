"""Saved public-recipe bridge; model/data hooks remain owned by the recipe."""

import json
import os
import subprocess
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from orchestration.lifecycle import Lifecycle
from orchestration.nsys import NsysCapture
from orchestration.nsys_worker import reports
from orchestration.processes import distributed_command, worker_command

from .api import run, validate
from .monitor import GpuMonitor


def benchmark(
    run_dir,
    *,
    output_dir,
    steps=7,
    warmup_steps=1,
    units_per_step=None,
    unit_label="items",
    monitor=True,
    trace_final_step=True,
    nsys=None,
):
    """Launch an existing public training run's selected Plan on one node.

    Ranks, initialization, input geometry, optimizer schedules and runtime
    options come from the saved run. No build/search/calibration is performed.
    This benchmark always disables evaluation, checkpoint writes and resume.
    ``output_dir`` must be new; the original training run is read-only.
    ``nsys=NsysCapture(...)`` launches one collector per global rank, recording
    metrics on that rank's GPU and saving ``profiling/rank-NNNNN/trace.nsys-rep``.
    ``NsysCapture(output_dir=...)`` overrides that common capture directory.
    """
    validate(steps, warmup_steps, units_per_step, trace_final_step, nsys)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("benchmark launches workers itself; use run inside torchrun")
    source = Path(run_dir).resolve()
    for name in ("config.json", "model.json", "selected.json", "plans/training.ssp"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    options = json.loads((source / "config.json").read_text())
    root = Path(output_dir).resolve()
    if nsys is not None:
        capture_dir = (
            Path(nsys.output_dir).expanduser().resolve()
            if nsys.output_dir is not None
            else root / "profiling"
        )
        command = distributed_command(
            nsys.rank_command(
                worker_command("benchmarking.plan_runner.recipe", [root]),
                output_dir=capture_dir,
            ),
            ranks=options["ranks"],
        )
    else:
        command = worker_command(
            "benchmarking.plan_runner.recipe", [root], ranks=options["ranks"]
        )
    root.mkdir(parents=True, exist_ok=False)
    request = dict(
        run_dir=str(source),
        steps=steps,
        warmup_steps=warmup_steps,
        units_per_step=units_per_step,
        unit_label=unit_label,
        trace_final_step=trace_final_step,
        nsys=None if nsys is None else asdict(nsys),
    )
    (root / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (root / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else list(map(str, range(options["ranks"])))
    with (
        Lifecycle(root / "launcher.jsonl") as timing,
        (
            GpuMonitor(root, devices=",".join(devices[: options["ranks"]]))
            if monitor
            else nullcontext()
        ),
        timing.phase("workers"),
        (root / "workers.log").open("x") as log,
    ):
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        timing.event("workers.returned", returncode=result.returncode)
    if result.returncode:
        tail = "\n".join((root / "workers.log").read_text().splitlines()[-20:])
        raise RuntimeError(
            f"saved-Plan benchmark failed; see {root / 'workers.log'}\n{tail}"
        )
    result = json.loads((root / "result.json").read_text())
    if nsys is not None:
        captures = reports(capture_dir, ranks=options["ranks"])
        for capture in captures:
            path = capture_dir / capture["report_path"]
            capture["report_path"] = str(
                path.relative_to(root) if path.is_relative_to(root) else path
            )
        result["nsys"] = dict(
            **asdict(nsys),
            reports=captures,
            # A real report is required here. Inspect its ranges, kernels and
            # device-metric samples offline; a requested flag is not evidence.
            content_verified=False,
        )
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def worker(directory):
    root = Path(directory)
    request = json.loads((root / "request.json").read_text())
    source = Path(request.pop("run_dir"))
    options = json.loads((source / "config.json").read_text())
    options.update(json.loads((source / "selected.json").read_text()))
    metadata = json.loads((source / "model.json").read_text())
    options.update(
        steps=request["steps"], eval_every=0, checkpoint_every=0, resume=None
    )
    # A saved recipe's old diagnostic window must not silently profile this
    # benchmark. Capture belongs to this request, with matching timing exclusions.
    if "nsys_steps" in options:
        options["nsys_steps"] = 0
    if request.get("nsys") is not None:
        request["nsys"] = NsysCapture(**request["nsys"])
    from workloads.recipes.training.configuration import adapter

    definition = adapter(options["recipe"])
    # The adapter's model-specific initializer is entered only inside run(),
    # after the Executor installs its allocator. No setup/build hook is called.
    return run(
        source / "plans/training.ssp",
        output_dir=root,
        monitor=False,
        **request,
        initializer=lambda context: definition.initializer(
            options, metadata, rank=context.rank
        ),
        inputs=lambda context: definition.inputs(options, rank=context.rank),
        trainer_options=lambda context: definition.trainer_options(options, metadata),
    )


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
