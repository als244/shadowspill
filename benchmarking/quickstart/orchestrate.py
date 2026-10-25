"""A process boundary between preparation and the execution allocator."""

import os
import subprocess
import sys
from pathlib import Path

from .phases.common import read, reference, write
from .results import write_measurements


def run(
    factory,
    *,
    search_budget_gib=(),
    spill_gib=0,
    output_dir,
    run_budget_gib=None,
    steps=5,
    device="cuda:0",
    factory_args=None,
    communication_info=None,
    warmup_seconds=0.05,
    measure_seconds=0.18,
    search_workers=1,
    external_headroom_gib=0.5,
    plots=True,
    phase="all",
):
    """Compose public phases; factories provide metadata, inputs and a bounded loader.

    Factory and its keyword arguments must be importable/JSON-serializable so
    workers can reconstruct setup independently. See the phase quickstart guide.
    Each GPU phase has its own process; ordinary tensor allocation in the caller
    cannot contaminate the execution allocator.
    """
    root = Path(output_dir).resolve()
    if (root / "request.json").exists():
        raise FileExistsError("use a fresh output directory or resume one named phase")
    communication = None
    if communication_info is not None:
        communication = str(root / "supplied-communication.ssp")
        communication_info.save(communication)
    request = {
        "factory": reference(factory),
        "factory_args": dict(factory_args or {}),
        "output_dir": str(root),
        "device": "cuda:0" if str(device) == "cuda" else str(device),
        "search_budgets": [int(v * (1 << 30)) for v in search_budget_gib],
        "run_budgets": [
            int(v * (1 << 30))
            for v in (search_budget_gib if run_budget_gib is None else run_budget_gib)
        ],
        "spill_bytes": int(spill_gib * (1 << 30)),
        "steps": steps,
        "communication_info": communication,
        "warmup_seconds": warmup_seconds,
        "measure_seconds": measure_seconds,
        "search_workers": search_workers,
        "plots": plots,
        "external_headroom_bytes": int(external_headroom_gib * (1 << 30)),
    }
    return run_request(request, phase)


def run_request(request, phase):
    _validate(request, phase)
    root = Path(request["output_dir"])
    request_file = root / "request.json"
    write(request_file, request)
    for item in ("build", "profile", "plan"):
        if phase in ("all", item):
            _launch(item, request_file)
    if phase in ("all", "execute"):
        winners = read(root / "plans" / "index.json")
        selected = {entry["budget"]: index for index, entry in enumerate(winners)}
        for budget in request["run_budgets"]:
            if budget not in selected:
                print(
                    f"execute: skipping infeasible budget {budget / (1 << 30):g} GiB",
                    flush=True,
                )
                continue
            _launch("execute", request_file, selection=selected[budget])
        write_measurements(root, winners, plots=request["plots"])
    print(f"quickstart {phase} complete: {root}", flush=True)
    return 0


def _validate(request, phase):
    if phase not in {"all", "build", "profile", "plan", "execute"}:
        raise ValueError("unknown quickstart phase")
    if phase in {"all", "plan"} and (
        not request["search_budgets"]
        or any(b <= 0 for b in request["search_budgets"])
        or request["spill_bytes"] <= 0
    ):
        raise ValueError("search budgets must be positive")
    if type(request["steps"]) is not int or request["steps"] < 1:
        raise ValueError("steps must be positive")
    if not set(request["run_budgets"]) <= set(request["search_budgets"]):
        raise ValueError("run budgets must be search budgets")
    if request["warmup_seconds"] < 0 or request["measure_seconds"] <= 0:
        raise ValueError("invalid measurement durations")
    if not request["device"].startswith("cuda:") or not request["device"][5:].isdigit():
        raise ValueError("device must name a CUDA ordinal, for example cuda:0")


def _launch(phase, request_file, selection=None):
    suffix = "" if selection is None else f"-{selection:04d}"
    log_path = request_file.parent / "logs" / (phase + suffix + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        "-u",
        "-m",
        "benchmarking.quickstart.phases.worker",
        phase,
        str(request_file),
    ]
    if selection is not None:
        command += ["--selection", str(selection)]
    environment = dict(os.environ)
    if phase == "plan":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    print(f"{phase}{suffix}: starting; log {log_path}", flush=True)
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as worker,
    ):
        try:
            for line in worker.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = worker.wait()
        except BaseException:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            raise
    if code:
        raise RuntimeError(f"quickstart {phase} failed ({code}); see {log_path}")
